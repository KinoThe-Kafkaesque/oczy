"""HF-substrate KV-slot fact injection experiment (Sprint 1 / S1.3).

Implements conditions C0–C3 and secondary analyses from
``research/09-hf-kv-slot-fact-injection.md`` using the HFDriver API.

Pre-registered 2026-07-01 — this module MUST NOT deviate from the spec.
Deviations are reported explicitly in the log.

Conditions (all greedy/deterministic; 3 facts from lane_02):
  - C0: probe alone, no injection (expected: rank far from 1)
  - C1: fact text prefixed to probe prompt (upper anchor)
  - C2: encode_kv + token_ranks_with_kv (fact NOT in visible prompt)
  - C3: cvec-only at legacy working amplitude (known-fail anchor)

Primary metric: ``hf_kv_slot_rank1_count`` (C2 facts at rank 1).

Secondary (exploratory):
  - splice position variants (front vs. immediately pre-blank)
  - K/V norm scaling (0.5x, 1x, 2x)
  - injection latency (ms)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from oczy.experiments.multi_fact_stressor import FACTS, QUERIES, TARGETS
from oczy.lm.hf_driver import HFDriver, KVHandle
from oczy.lm.hf_model_choice import HF_MODEL_ID

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Probe template from lanes/lane_02.py — single fixed form for all 3 facts.
# The lowercase nudge reduces surface-form ambiguity so the natural
# continuation is the lowercase target token.
_PROBE_TEMPLATE = "\n\nRecall the answer in lowercase. Question: {}\nAnswer:"

# Legacy working amplitude from CortexAgentConfig.articulate_scale
# (LFM2.5-1.2B-Instruct Q4_K_M, steering_mode="raw_hidden").
# C3 is expected to fail — this is a reference anchor, not a tuned value.
LEGACY_CVEC_AMPLITUDE = 0.001

# Number of facts probed (first N from multi_fact_stressor).
_PROBE_SIZE = 3

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ConditionResult:
    """One fact × condition measurement."""

    condition: str
    fact_idx: int
    fact: str
    query: str
    target: str
    rank: int
    top1: str
    target_id: int | None = None
    latency_ms: float = 0.0

    def rank1(self) -> bool:
        return self.rank == 0  # rank=0 means argmax


@dataclass
class ExperimentResult:
    """Aggregate result for the full experiment."""

    model_id: str
    results: list[ConditionResult] = field(default_factory=list)

    def _by_condition(self, condition: str) -> list[ConditionResult]:
        return [r for r in self.results if r.condition == condition]

    @property
    def hf_kv_slot_rank1_count(self) -> int:
        """Primary metric: C2 facts at rank 1 (rank=0, i.e. argmax)."""
        return sum(1 for r in self._by_condition("C2") if r.rank1())

    @property
    def c0_ranks(self) -> list[int]:
        """Sanity guard: C0 ranks must be far from 1 for accepted facts."""
        return [r.rank for r in self._by_condition("C0")]

    def rank_table(self) -> str:
        """Multi-line markdown table: fact x condition ranks."""
        lines = ["| Fact | Target | C0 | C1 | C2 | C3 |",
                 "|---|---:|---:|---:|---:|---:|"]
        for i in range(_PROBE_SIZE):
            c0 = self._find("C0", i)
            c1 = self._find("C1", i)
            c2 = self._find("C2", i)
            c3 = self._find("C3", i)
            label = self._fact_label(i)
            lines.append(
                f"| {label} | {c0.target} | {c0.rank} | {c1.rank} | "
                f"{c2.rank} | {c3.rank} |"
            )
        return "\n".join(lines)

    def top1_table(self) -> str:
        """Multi-line markdown table: fact x condition top-1 tokens."""
        lines = ["| Fact | Target | C0 top1 | C1 top1 | C2 top1 | C3 top1 |",
                 "|---|---|---|---|---|---|"]
        for i in range(_PROBE_SIZE):
            c0 = self._find("C0", i)
            c1 = self._find("C1", i)
            c2 = self._find("C2", i)
            c3 = self._find("C3", i)
            label = self._fact_label(i)
            lines.append(
                f"| {label} | {c0.target} | `{c0.top1}` | `{c1.top1}` | "
                f"`{c2.top1}` | `{c3.top1}` |"
            )
        return "\n".join(lines)

    @staticmethod
    def _fact_label(idx: int) -> str:
        labels = ["alpha/skylark", "beta/rook", "level7/marmalade"]
        return labels[idx]

    def _find(self, condition: str, fact_idx: int) -> ConditionResult:
        for r in self.results:
            if r.condition == condition and r.fact_idx == fact_idx:
                return r
        raise KeyError(f"{condition}/{fact_idx}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _target_token(driver: HFDriver, target: str) -> str:
    """Return the space-prefixed target string for tokenization.

    Lane_02 / legacy llama.cpp convention: targets are tokenized with a
    leading space (``" " + target``) to match the natural continuation
    token after ``"Answer:"`` or similar prompt endings.  This ensures
    the token ID matches what the model would actually emit as the next
    token.
    """
    return " " + target


def _probe_prompt(query: str) -> str:
    return _PROBE_TEMPLATE.format(query)


def _compute_cvec_from_fact(driver: HFDriver, fact: str) -> np.ndarray:
    """Compute a steering vector direction from the fact's last-token hidden.

    Uses the raw_hidden method: peek the last-token embedding of the fact
    text, L2-normalize to get a unit direction vector.  This is the same
    method KVCortex uses in steering_mode="raw_hidden".
    """
    emb = driver.peek_embedding(fact, last_token_only=True)
    norm = float(np.linalg.norm(emb))
    if norm > 0:
        emb = emb / norm
    return emb.astype(np.float32)


def _scale_kv_handle(handle: KVHandle, k_scale: float, v_scale: float) -> KVHandle:
    """Return a new KVHandle with K and V tensors scaled element-wise.

    Does NOT mutate the original handle — returns a deep copy.
    Handles both legacy tuple-of-tuples and newer DynamicCache formats.
    """
    from transformers.cache_utils import DynamicCache

    pkv = handle.past_key_values

    if isinstance(pkv, DynamicCache):
        new_cache = DynamicCache()
        for layer_idx, layer in enumerate(pkv):
            # Each layer is a tuple (k, v, ...) where the extra elements
            # (like cache_position) are None or metadata.
            k = layer[0]
            v = layer[1]
            new_cache.update(
                k * k_scale, v * v_scale, layer_idx=layer_idx
            )
        return KVHandle(past_key_values=new_cache, seq_len=handle.seq_len)

    # Legacy tuple-of-tuples path (older transformers versions).
    new_pkv = tuple(
        tuple(
            (t[0] * k_scale, t[1] * v_scale)
            if isinstance(t, tuple) and len(t) >= 2
            else t
        )
        for t in pkv
    )
    return KVHandle(past_key_values=new_pkv, seq_len=handle.seq_len)


def _token_ranks_with_kv_splice(
    driver: HFDriver,
    prompt_before: str,
    prompt_after: str,
    handle: KVHandle,
    targets: list[str],
) -> list[dict[str, Any]]:
    """Run token_ranks with a KV handle spliced between two prompt parts.

    The KV handle is positioned between *prompt_before* and *prompt_after*,
    so the model attends to:
        [handle KVs] [prompt_before tokens] [prompt_after tokens]
    and we read logits at the final position (after prompt_after).
    """
    # Encode the prefix portion with the KV handle.
    prefix_ids = driver._tokenize(prompt_before)  # type: ignore[attr-defined]
    prefix_len = prefix_ids.shape[1]

    prefix_pos = torch.arange(
        handle.seq_len, handle.seq_len + prefix_len, dtype=torch.long
    ).unsqueeze(0)

    out = driver._model(  # type: ignore[attr-defined]
        input_ids=prefix_ids,
        past_key_values=handle.past_key_values,
        position_ids=prefix_pos,
        use_cache=True,
    )

    # Encode the suffix portion on top of the combined KV.
    suffix_ids = driver._tokenize(prompt_after)  # type: ignore[attr-defined]
    suffix_len = suffix_ids.shape[1]

    suffix_pos = torch.arange(
        handle.seq_len + prefix_len,
        handle.seq_len + prefix_len + suffix_len,
        dtype=torch.long,
    ).unsqueeze(0)

    out = driver._model(  # type: ignore[attr-defined]
        input_ids=suffix_ids,
        past_key_values=out.past_key_values,
        position_ids=suffix_pos,
    )

    logits = out.logits[0, -1, :].cpu()

    results: list[dict[str, Any]] = []
    for target in targets:
        target_ids = driver._tokenizer.encode(  # type: ignore[attr-defined]
            target, add_special_tokens=False
        )
        if not target_ids:
            results.append(
                {"target": target, "rank": driver.n_vocab, "top1": ""}  # type: ignore[attr-defined]
            )
            continue
        tid = target_ids[0]
        rank = int((logits > logits[tid]).sum().item())
        top1_id = int(logits.argmax().item())
        top1 = driver._tokenizer.decode([top1_id])  # type: ignore[attr-defined]
        results.append(
            {"target": target, "rank": rank, "top1": top1, "target_id": tid}
        )
    return results


# ---------------------------------------------------------------------------
# Primary conditions C0–C3
# ---------------------------------------------------------------------------


def _run_condition_c0(
    driver: HFDriver, fact_idx: int, fact: str, query: str, target: str
) -> ConditionResult:
    """C0: probe alone, no injection. Expected: rank far from 1."""
    driver.clear_cvec()
    prompt = _probe_prompt(query)
    ranks = driver.token_ranks(prompt, [_target_token(driver, target)])
    r = ranks[0]
    return ConditionResult(
        condition="C0",
        fact_idx=fact_idx,
        fact=fact,
        query=query,
        target=target,
        rank=r["rank"],
        top1=r["top1"],
        target_id=r.get("target_id"),
    )


def _run_condition_c1(
    driver: HFDriver, fact_idx: int, fact: str, query: str, target: str
) -> ConditionResult:
    """C1: fact text prefixed to probe prompt. Upper anchor (expected rank 1)."""
    driver.clear_cvec()
    prompt = fact + _probe_prompt(query)
    ranks = driver.token_ranks(prompt, [_target_token(driver, target)])
    r = ranks[0]
    return ConditionResult(
        condition="C1",
        fact_idx=fact_idx,
        fact=fact,
        query=query,
        target=target,
        rank=r["rank"],
        top1=r["top1"],
        target_id=r.get("target_id"),
    )


def _run_condition_c2(
    driver: HFDriver, fact_idx: int, fact: str, query: str, target: str
) -> ConditionResult:
    """C2: encode_kv + token_ranks_with_kv — fact NOT in visible prompt."""
    driver.clear_cvec()
    t0 = time.perf_counter()
    handle = driver.encode_kv(fact)
    prompt = _probe_prompt(query)
    ranks = driver.token_ranks_with_kv(prompt, handle, [_target_token(driver, target)])
    latency_ms = (time.perf_counter() - t0) * 1000.0
    r = ranks[0]
    return ConditionResult(
        condition="C2",
        fact_idx=fact_idx,
        fact=fact,
        query=query,
        target=target,
        rank=r["rank"],
        top1=r["top1"],
        target_id=r.get("target_id"),
        latency_ms=latency_ms,
    )


def _run_condition_c3(
    driver: HFDriver, fact_idx: int, fact: str, query: str, target: str
) -> ConditionResult:
    """C3: cvec-only at legacy working amplitude. Known-fail anchor."""
    driver.clear_cvec()
    cvec = _compute_cvec_from_fact(driver, fact)
    driver.set_cvec_uniform(cvec, scale=LEGACY_CVEC_AMPLITUDE)
    prompt = _probe_prompt(query)
    ranks = driver.token_ranks(prompt, [_target_token(driver, target)])
    driver.clear_cvec()
    r = ranks[0]
    return ConditionResult(
        condition="C3",
        fact_idx=fact_idx,
        fact=fact,
        query=query,
        target=target,
        rank=r["rank"],
        top1=r["top1"],
        target_id=r.get("target_id"),
    )


# ---------------------------------------------------------------------------
# Secondary analyses (exploratory)
# ---------------------------------------------------------------------------


def _run_splice_position_variants(
    driver: HFDriver, fact_idx: int, fact: str, query: str, target: str
) -> list[ConditionResult]:
    """Splice position: front vs. immediately pre-blank.

    "Front" is the C2 condition (all KV entries before the entire probe).
    "Immediately pre-blank" inserts the fact KV right before "Answer:",
    as close to the generation position as possible.
    """
    driver.clear_cvec()
    handle = driver.encode_kv(fact)

    # Build prompt parts for pre-blank splice.
    # Template: "\n\nRecall the answer in lowercase. Question: {query}\nAnswer:"
    # Split before "Answer:" so the fact KV is as close to blank as possible.
    prompt_before = f"\n\nRecall the answer in lowercase. Question: {query}\n"
    prompt_after = "Answer:"
    full_prompt = _probe_prompt(query)

    results: list[ConditionResult] = []

    # Front (C2-equivalent — fact KV before everything).
    ranks_front = driver.token_ranks_with_kv(full_prompt, handle, [_target_token(driver, target)])
    r = ranks_front[0]
    results.append(
        ConditionResult(
            condition="C2-splice-front",
            fact_idx=fact_idx,
            fact=fact,
            query=query,
            target=target,
            rank=r["rank"],
            top1=r["top1"],
            target_id=r.get("target_id"),
        )
    )

    # Immediately pre-blank (fact KV right before "Answer:").
    ranks_pre = _token_ranks_with_kv_splice(
        driver, prompt_before, prompt_after, handle, [_target_token(driver, target)]
    )
    r = ranks_pre[0]
    results.append(
        ConditionResult(
            condition="C2-splice-preblank",
            fact_idx=fact_idx,
            fact=fact,
            query=query,
            target=target,
            rank=r["rank"],
            top1=r["top1"],
            target_id=r.get("target_id"),
        )
    )

    return results


def _run_kv_norm_variants(
    driver: HFDriver, fact_idx: int, fact: str, query: str, target: str
) -> list[ConditionResult]:
    """K/V norm scaling: 0.5x, 1x, 2x on the C2 path."""
    driver.clear_cvec()
    handle = driver.encode_kv(fact)
    prompt = _probe_prompt(query)

    results: list[ConditionResult] = []
    for scale in (0.5, 1.0, 2.0):
        scaled_handle = _scale_kv_handle(handle, scale, scale)
        label = f"C2-scale-{scale:.1f}x"
        ranks = driver.token_ranks_with_kv(prompt, scaled_handle, [_target_token(driver, target)])
        r = ranks[0]
        results.append(
            ConditionResult(
                condition=label,
                fact_idx=fact_idx,
                fact=fact,
                query=query,
                target=target,
                rank=r["rank"],
                top1=r["top1"],
                target_id=r.get("target_id"),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    driver: HFDriver,
    include_secondary: bool = True,
) -> ExperimentResult:
    """Run C0–C3 and optional secondary analyses on all facts.

    Args:
        driver: An already-loaded HFDriver instance.
        include_secondary: Whether to include splice-position and
            K/V scaling secondary analyses.

    Returns:
        ``ExperimentResult`` with all condition measurements.
    """
    facts = list(FACTS[:_PROBE_SIZE])
    queries = list(QUERIES[:_PROBE_SIZE])
    targets = list(TARGETS[:_PROBE_SIZE])

    result = ExperimentResult(model_id=driver.model_id)

    for i, (fact, query, target) in enumerate(
        zip(facts, queries, targets, strict=True)
    ):
        # Primary conditions
        result.results.append(
            _run_condition_c0(driver, i, fact, query, target)
        )
        result.results.append(
            _run_condition_c1(driver, i, fact, query, target)
        )
        result.results.append(
            _run_condition_c2(driver, i, fact, query, target)
        )
        result.results.append(
            _run_condition_c3(driver, i, fact, query, target)
        )

        # Secondary (exploratory)
        if include_secondary:
            result.results.extend(
                _run_splice_position_variants(driver, i, fact, query, target)
            )
            result.results.extend(
                _run_kv_norm_variants(driver, i, fact, query, target)
            )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _emit_report(result: ExperimentResult) -> str:
    """Produce a full text report in the experiments_logs format."""
    lines: list[str] = []

    lines.append("# S1.3 — HF-substrate KV-slot fact injection")
    lines.append("")
    lines.append("## Date: 2026-07-02")
    lines.append("")
    lines.append(f"## Model: {result.model_id}")
    lines.append("")
    lines.append("## Spec")
    lines.append("")
    lines.append("Pre-registered: `research/09-hf-kv-slot-fact-injection.md`")
    lines.append("")

    # Rank table
    lines.append("## Rank table (fact x condition)")
    lines.append("")
    lines.append(result.rank_table())
    lines.append("")

    # Top-1 table
    lines.append("## Top-1 token table")
    lines.append("")
    lines.append(result.top1_table())
    lines.append("")

    # Primary metric
    c2_count = result.hf_kv_slot_rank1_count
    c0_ranks = result.c0_ranks
    c0_ok = all(r > 10 for r in c0_ranks)

    lines.append("## Primary metric")
    lines.append("")
    lines.append(f"`hf_kv_slot_rank1_count` = **{c2_count}** / 3")
    lines.append("")

    if c2_count >= 2 and c0_ok:
        lines.append("**Verdict: ACCEPT H-KV** — the hypothesis survives.")
    else:
        lines.append("**Verdict: REFUTE H-KV** — the hypothesis does not survive.")
        if c2_count < 2:
            lines.append(
                f"  Reason: C2 rank-1 count ({c2_count}) < 2 (threshold)."
            )
        if not c0_ok:
            lines.append(
                f"  Reason: C0 sanity guard failed — some facts already "
                f"at low rank without injection (C0 ranks: {c0_ranks})."
            )
    lines.append("")

    # Sanity guard
    lines.append("## Sanity guard: C0 baseline")
    lines.append("")
    lines.append(f"C0 ranks: {c0_ranks}")
    lines.append(
        f"C0 far-from-1 check: {'PASS' if c0_ok else 'FAIL'} "
        f"(all ranks > 10)"
    )
    lines.append("")

    # C2 latency
    c2_latencies = [
        r.latency_ms
        for r in result.results
        if r.condition == "C2" and r.latency_ms > 0
    ]
    if c2_latencies:
        lines.append("## C2 injection latency (ms)")
        lines.append("")
        lines.append(f"Mean: {np.mean(c2_latencies):.2f} ms")
        lines.append(f"Median: {np.median(c2_latencies):.2f} ms")
        lines.append(f"Range: {min(c2_latencies):.2f} – {max(c2_latencies):.2f} ms")
        lines.append("")

    # Secondary: splice position
    splice_front = [r for r in result.results if r.condition == "C2-splice-front"]
    splice_pre = [r for r in result.results if r.condition == "C2-splice-preblank"]
    if splice_front or splice_pre:
        lines.append("## Secondary: splice position (exploratory)")
        lines.append("")
        lines.append("| Fact | Front (C2) | Pre-blank |")
        lines.append("|---|---:|--:|")
        for i in range(_PROBE_SIZE):
            sf = next((r for r in splice_front if r.fact_idx == i), None)
            sp = next((r for r in splice_pre if r.fact_idx == i), None)
            label = result._fact_label(i)
            lines.append(
                f"| {label} | {sf.rank if sf else '?'} | "
                f"{sp.rank if sp else '?'} |"
            )
        lines.append("")

    # Secondary: K/V norm scaling
    scale_results: dict[float, list[ConditionResult]] = {}
    for scale in (0.5, 1.0, 2.0):
        label = f"C2-scale-{scale:.1f}x"
        scale_results[scale] = [
            r for r in result.results if r.condition == label
        ]
    if any(scale_results.values()):
        lines.append("## Secondary: K/V norm scaling (exploratory)")
        lines.append("")
        lines.append("| Fact | 0.5x | 1.0x | 2.0x |")
        lines.append("|---|---:|---:|---:|")
        for i in range(_PROBE_SIZE):
            row = [result._fact_label(i)]
            for scale in (0.5, 1.0, 2.0):
                r = next(
                    (r for r in scale_results[scale] if r.fact_idx == i), None
                )
                row.append(str(r.rank) if r else "?")
            lines.append(f"| {' | '.join(row)} |")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    """Run the full experiment on HF_MODEL_ID and write the log."""
    import sys

    print("S1.3: Loading model...", file=sys.stderr)
    driver = HFDriver.load(HF_MODEL_ID)
    print(f"  Model: {driver.model_id}", file=sys.stderr)
    print(f"  n_embd={driver.n_embd}  n_layers={driver.n_layers}  "
          f"n_vocab={driver.n_vocab}", file=sys.stderr)

    try:
        print("S1.3: Running experiment...", file=sys.stderr)
        result = run_experiment(driver, include_secondary=True)

        report = _emit_report(result)
        print(report)

        # Write log file
        log_path = "experiments_logs/2026-07-01_s1_3_hf_kv_slot_injection.md"
        with open(log_path, "w") as f:
            f.write(report)
        print(f"\nLog written to {log_path}", file=sys.stderr)

        # Brief terminal summary
        verdict = (
            "ACCEPT H-KV"
            if result.hf_kv_slot_rank1_count >= 2
            and all(r > 10 for r in result.c0_ranks)
            else "REFUTE H-KV"
        )
        print(
            f"\nS1.3 complete. {verdict}. "
            f"hf_kv_slot_rank1_count={result.hf_kv_slot_rank1_count}/3",
            file=sys.stderr,
        )
    finally:
        driver.close()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
