# Oczy Experiments

Concrete, runnable experiment designs that operationalize the
[research agenda](../research/README.md). One directory per project; each
`README.md` is a self-contained spec: objective, setup, a matched-pair ablation
matrix, procedure, **exact** metric definitions, acceptance **and kill**
criteria, expected failure modes, and the artifacts (new `src/oczy/experiments/`
modules) to add.

Experiments 01–07 are now **implemented and tested** (modules under
`src/oczy/experiments/`); 08 and 09 remain specifications only. Each `README.md`
is a self-contained spec: objective, setup, a matched-pair ablation matrix,
procedure, **exact** metric definitions, acceptance **and kill** criteria,
expected failure modes, and the artifacts to add. See each experiment's
status block (near the top) for its current implementation and scientific
verdict, and the [campaign 0d48130 evidence log](../experiments_logs/2026-07-11_campaign_0d48130.md)
for the curated run results.

## Index

| # | Experiment | One question it answers | Status | Module |
|---|------------|-------------------------|--------|--------|
| [01](01-correction-to-competence-benchmark/) | Correction-to-Competence Benchmark v2 | Can a behavior-only scorecard separate architectures the current eval ties at 1.0? | implemented/tested — **NULL** | `oczy.experiments.correction_competence_v2` |
| [02](02-kv-slot-fact-injection/) | KV-slot fact injection | Can a reserved `(k,v)` slot force an exact token that a residual cvec provably cannot? | implemented/tested — **REFUTED** | `oczy.experiments.kv_slot_injection` |
| [03](03-layer-l-hidden-extraction/) | Layer-L hidden extraction | Does feeding the cortex a real mid-layer residual change `warm_state` trajectories vs the final-layer mean-pool? | implemented — **infrastructure-blocked** (S1.4 REFUTED) | `oczy.experiments.layer_l_probe` |
| [04](04-context-scoped-attractors/) | Context-scoped attractors | Can two senses of one token coexist as distinct basins without obliterating each other? | implemented/tested — **POSITIVE** | `oczy.experiments.scope_selectivity_stressor` |
| [05](05-metabolism-loop-closure/) | Metabolism loop closure | Do repeated corrections *compound* cold-state drift, and does that drift (not a label) drive the answer? | implemented/tested — **NULL** | `oczy.experiments.metabolism_loop` |
| [06](06-bounded-growth-consolidation/) | Bounded-growth consolidation | Can a trained encoder + hypernetwork raise `behavior_delta_per_byte` vs the random-projection baseline? | implemented/tested — **POSITIVE** (5-seed) | `oczy.experiments.bounded_growth.bounded_growth_eval` |
| [07](07-conversation-world-model-rl/) | Conversation world model (RL Phase 0) | Can a self-supervised model predict acceptance / correction-type before answering, beating the lexical stop-gap? | implemented/tested — **POSITIVE** (marker-free) + **NULL** (critic) | `oczy.experiments.conversation_world_model` |
| [08](08-oczy-pi-tool-calling-curriculum/) | Oczy Pi tool-calling curriculum | Can the plastic cortex teach a frozen 1.2B model to use Pi's tools (read/bash/write/edit) across multi-turn agentic tasks? | **unimplemented** (spec only) | `oczy.experiments.tool_calling_curriculum` |
| [09](09-meta-trained-cortex-frozen-language-organ/) | Meta-trained cortex over a frozen language organ | Can a cortex learn a reusable write/read/consolidate rule, then learn an unseen behavior without retrieval or online backprop? | **unimplemented** (spec only) | `oczy.experiments.meta_cortex.run_meta_test` |

## How to run

All experiments use `uv` and reuse existing scaffolds rather than inventing
infrastructure. The canonical loop is:

```bash
# from repo root
uv run python -m oczy.experiments.<module> [--driver mock|real] [flags]
```

- **`--driver mock`** runs against `_MockDriver` (fast, deterministic, *no
  semantics*) — use it to validate the harness and as the structural-null
  control. Mock exact-recall is structurally 0; that is expected, not a bug.
- **`--driver real`** loads `LiquidAI/LFM2.5-1.2B-Instruct-Q4_K_M.gguf`
  (cached under `~/.cache/huggingface/hub/`) — the semantic test. Slower
  (seconds per turn, CPU).
- Each spec emits `METRIC ...` / `ASI ...` lines compatible with the
  autoresearch harness, and writes a JSON report under `reports/`.
- Experiment **09** uses the frozen HF `Qwen2.5-0.5B-Instruct` organ and a
  separately manifest-frozen `meta_cortex/v1` instrument. Its primary condition
  explicitly disables retrieval and permits only cortex fast/slow state to
  change during meta-test.

The headline regression gate stays `bash autoresearch.sh`
(`code_qa_accuracy` must remain 1.0) — but note that this very saturation is
what experiment **01** exists to fix.

## Reusable scaffolds these specs build on

| Scaffold | What it provides |
|----------|------------------|
| `src/oczy/experiments/cortex_agent.py` | `CortexAgent` — perceive / metabolize / articulate / consolidate; the organism glue. |
| `src/oczy/experiments/ingestion.py` | Chunker → salience → embedder → aggregator pipeline + `TurnDigest`. |
| `src/oczy/experiments/multi_fact_stressor.py` | Real/mock driver loaders, needle/multi-fact probes, `METRIC`/`ASI` print lines. |
| `src/oczy/experiments/needle_sweep.py` | Position/length needle-recall sweeps. |
| `src/oczy/experiments/eval_suite.py` | Snapshot/score scaffold + 7-metric scorecard (the one 01 rebuilds). |
| `src/oczy/experiments/smoke_consolidation_uptake_compare.py` | The correction loop + SVD-init + hard consolidate + domain-word scoring. |
| `src/oczy/experiments/organism_curriculum/` | Stages 0–5, `scoring.py` (`probe_matches`), `run_curriculum.py`. |
| `plastic-cortex/.../kv_cortex.py` | `KVCortex` — warm/cold state, `observe`, `emit_cvec`, `consolidate`, SVD init. |
| `world-model-critic`, `neural-hippocampus`, `identity-hypernetwork`, `experience-autoencoder`, `skill-immune-cortex` | The five metabolism organs. |

## Status

Experiments 01–07 are implemented and have been run under Campaign 0d48130
(2026-07-11). Scientific outcomes from that campaign:

| # | Outcome | Primary metric |
|---|---------|----------------|
| 01 | **NULL** (behavior-delta transfer) | `v2_behavior_delta_mock=0.0` |
| 02 | **REFUTED** (KV-slot injection) | `kv_slot_rank1_count=0.0` |
| 03 | **Infrastructure blocked** — no verdict; authoritative pre-campaign verdict is S1.4 **REFUTED** | no metrics (HF snapshot transfer failures) |
| 04 | **POSITIVE** (scope selectivity) | `scope_selectivity_index=1.0` |
| 05 | **NULL** (metabolism drift) | `metabolism_drift_delta=0.0` |
| 06 | **POSITIVE** (bounded growth, 5 seeds) | `bounded_growth_m1_ratio=0.002079` |
| 07 | **POSITIVE** (marker-free uptake) + **NULL** (critic AUC) | `marker_free_uptake_gap=1.0`, `critic_auc_delta=0.0` |

Curated evidence: [`experiments_logs/2026-07-11_campaign_0d48130.md`](../experiments_logs/2026-07-11_campaign_0d48130.md).
Authoritative ledger: [`experiments_logs/LEDGER.md`](../experiments_logs/LEDGER.md).

Experiments **08** and **09** remain unimplemented specifications. The current
cortex sequence is **19 → 20 → 21** in the research agenda: Experiment **09**
operationalizes Research/20, while Experiment **08** remains an external
tool-use battery until the core frozen-organ cortex condition succeeds. See
each `research/NN-*.md` for verdict gates and each experiment README for its
build-out.
