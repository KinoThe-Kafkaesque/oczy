# S2.5 Forgetting Test — BLOCKED

**Date:** 2026-07-02
**Spec:** research/13-s2-forgetting-test.md
**Model:** Qwen/Qwen2.5-0.5B-Instruct (CPU float32, greedy)
**Stage:** Stage 0: Sense grounding (stage_0_grounding.json)
**Seeds:** 5
**Command:** `uv run python -m oczy.experiments.minimal_loop_forgetting --seeds 5 --stage 0`

## Blocking Cause

The pre-registered split `split_probes(stage, fraction=0.3, salt="v2")` produced **0 holdout probes** from stage 0's 8 total probes. All 8 probes hashed to the dev partition:

| Probe ID | Hash value | Partition |
|---|---|---|
| s0_log\|Show the log.\|retention | 0.4670 | dev |
| s0_file\|File the report.\|retention | 0.6429 | dev |
| s0_key\|Check the key.\|retention | 0.3944 | dev |
| s0_cell\|Edit the cell.\|retention | 0.5741 | dev |
| s0_record\|Play the record.\|retention | 0.5751 | dev |
| s0_branch\|Create a branch.\|retention | 0.5927 | dev |
| s0_model\|Deploy the model.\|retention | 0.5765 | dev |
| s0_run\|Start the run.\|retention | 0.4410 | dev |

This is a deterministic outcome of the SHA-256-based split at `fraction=0.3`. The split function's guarantee (force at least 1 holdout for stages with <4 probes) does not apply to stage 0 (8 probes). The probability of this outcome is (0.7)^8 ≈ 5.8%.

Since the S2.5 protocol scores holdout probes exclusively and there are 0, all per-arm accuracies are 0/0 = 0.0000, the validity gate `A_full − A_none ≥ 0.10` fails for every seed, and the experiment is **BLOCKED** — not a refutation of H-FORGET, but a data-availability block.

## 2×2 Arm Accuracies (mean ± 95% CI) — 0 holdout probes

| Arm | Raw traces | Consolidated artifact | Accuracy |
|---|---|---|---|
| **A_full** | kept | kept | 0.0000 ± 0.0000 (n=5) |
| **A_forget** | **deleted** | kept | 0.0000 ± 0.0000 (n=5) |
| **A_retrieval** | kept | **deleted** | 0.0000 ± 0.0000 (n=5) |
| **A_none** | **deleted** | **deleted** | 0.0000 ± 0.0000 (n=5) |

## Primary Metric

- **forgetting_survival_ratio:** BLOCKED (0 holdout probes → all seeds fail validity gate)

## Validity Gate

- `A_full − A_none ≥ 0.10`: 0/5 seeds passed (no holdout probes to score)
- BLOCKED seeds: 5/5

## Secondary Analyses

- **retrieval_dependence:** BLOCKED
- **behavior_delta_per_kb (pre-deletion):** 0.000000 ± 0.000000
- **behavior_delta_per_kb (post-deletion):** 0.000000 ± 0.000000

## Memory Footprint (per seed)

| Seed | Pre-deletion (bytes) | Post-deletion (bytes) | Delta |
|---|---|---|---|
| 0 | 1574 | 1413 | −161 |
| 1 | 1581 | 1413 | −168 |
| 2 | 1569 | 1413 | −156 |
| 3 | 1567 | 1413 | −154 |
| 4 | 1579 | 1413 | −166 |

Pre-deletion state includes: trained cortex state (cold/warm ~256 bytes each), hippocampus traces (~100-150 bytes), and consolidated articulation prefix text (~50-150 bytes depending on seed-shuffled summary phrasing). Post-deletion state has: zeroed cortex state and empty hippocampus (baseline driver overhead ~1400 bytes).

Deletion APIs verified: `delete_raw_traces()` zeroes hippocampus episode count and clears replay bank; `delete_consolidated_artifact()` clears prefix and resets cold/warm states to zero.

## Per-Seed Detail

| Seed | A_full | A_forget | A_retrieval | A_none | Survival | Retrieval | mem_pre | mem_post | Validity |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | BLOCKED | BLOCKED | 1574 | 1413 | BLOCKED |
| 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | BLOCKED | BLOCKED | 1581 | 1413 | BLOCKED |
| 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | BLOCKED | BLOCKED | 1569 | 1413 | BLOCKED |
| 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | BLOCKED | BLOCKED | 1567 | 1413 | BLOCKED |
| 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | BLOCKED | BLOCKED | 1579 | 1413 | BLOCKED |

## Implementation Notes

- Content channel: bounded 48-token consolidated articulation prefix (research/11)
- Hippocampus replay: `reinforce(query, k=1)` called per episode before `consolidate()` (replay_threshold=1)
- Cortex: `KVCortex` with d_cortex=128, observe layer=5
- Deletion APIs produce before/after `memory_bytes` pairs and verify post-conditions
- Snapshot/restore via `OrganismSnapshot` round-trips verified in unit tests (13/13 pass)
- No answer-time hippocampus access (verified by unit test tripwire)

## Remediation

To run this experiment meaningfully on stage 0, one of:
1. Use `salt="v3"` (or any other salt) to get a non-empty holdout split (violates pre-registration)
2. Use `fraction=0.5` to increase holdout probability (violates pre-registration)
3. Use a different stage (e.g., stage 1 has 1 holdout probe at `salt="v2"`, stage 2 has 3)
4. Add more probes to stage_0_grounding.json (requires eval version bump per AGENTS.md)

The experiment harness is functional and ready to run on any stage/split that yields ≥1 holdout probes.

## Verdict

**BLOCKED** — validity gate failed on all 5 seeds due to 0 holdout probes in the pre-registered split. Not a refutation of H-FORGET.

Per research/13 acceptance criteria:
- ACCEPT: forgetting_survival_ratio ≥ 0.8 with validity gate passing
- REFUTE: ratio < 0.5
- PARTIAL: 0.5 ≤ ratio < 0.8
- BLOCKED: validity gate failed on all seeds

---

## Addendum 2026-07-02 (post-adjudication)

The BLOCKED verdict above stands, but the binding reason has changed: the
0-holdout split it observed was an instrument failure, since repaired (see
the amendment in `research/13` and commit `11d8aca`). On the repaired split,
research/11 was adjudicated **REFUTE** (`2026-07-02_s2_1_minimal_loop.md`),
so the validity gate `A_full − A_none ≥ 0.10` cannot hold and **S2.5 remains
BLOCKED** until a loop design closes at C1. The deletion APIs, snapshot/
restore, and 2×2 harness (`minimal_loop_forgetting.py`) are merged and ready
for that day.
