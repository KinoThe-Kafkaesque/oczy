# Oczy Experiments

Concrete, runnable experiment designs that operationalize the
[research agenda](../research/README.md). One directory per project; each
`README.md` is a self-contained spec: objective, setup, a matched-pair ablation
matrix, procedure, **exact** metric definitions, acceptance **and kill**
criteria, expected failure modes, and the artifacts (new `src/oczy/experiments/`
modules) to add.

These are designs, not yet implementations — the `uv run python -m ...` commands
below name the module each spec asks you to create.

## Index

| # | Experiment | One question it answers | Sketch entry point |
|---|------------|-------------------------|--------------------|
| [01](01-correction-to-competence-benchmark/) | Correction-to-Competence Benchmark v2 | Can a behavior-only scorecard separate architectures the current eval ties at 1.0? | `oczy.experiments.correction_competence_v2` |
| [02](02-kv-slot-fact-injection/) | KV-slot fact injection | Can a reserved `(k,v)` slot force an exact token that a residual cvec provably cannot? | `oczy.experiments.kv_slot_injection` |
| [03](03-layer-l-hidden-extraction/) | Layer-L hidden extraction | Does feeding the cortex a real mid-layer residual change `warm_state` trajectories vs the final-layer mean-pool? | `oczy.experiments.layer_l_probe` |
| [04](04-context-scoped-attractors/) | Context-scoped attractors | Can two senses of one token coexist as distinct basins without obliterating each other? | `oczy.experiments.scope_selectivity_stressor` |
| [05](05-metabolism-loop-closure/) | Metabolism loop closure | Do repeated corrections *compound* cold-state drift, and does that drift (not a label) drive the answer? | `oczy.experiments.metabolism_loop` |
| [06](06-bounded-growth-consolidation/) | Bounded-growth consolidation | Can a trained encoder + hypernetwork raise `behavior_delta_per_byte` vs the random-projection baseline? | `oczy.experiments.bounded_growth.bounded_growth_eval` |
| [07](07-conversation-world-model-rl/) | Conversation world model (RL Phase 0) | Can a self-supervised model predict acceptance / correction-type before answering, beating the lexical stop-gap? | `oczy.experiments.conversation_world_model` |
| [08](08-oczy-pi-tool-calling-curriculum/) | Oczy Pi tool-calling curriculum | Can the plastic cortex teach a frozen 1.2B model to use Pi's tools (read/bash/write/edit) across multi-turn agentic tasks? | `oczy.experiments.tool_calling_curriculum` |
| [09](09-meta-trained-cortex-frozen-language-organ/) | Meta-trained cortex over a frozen language organ | Can a cortex learn a reusable write/read/consolidate rule, then learn an unseen behavior without retrieval or online backprop? | `oczy.experiments.meta_cortex.run_meta_test` |

## How to run (once implemented)

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

This directory now contains nine experiment specifications at different stages
of implementation. The current cortex sequence is **19 → 20 → 21** in the
research agenda: Experiment **09** operationalizes Research/20, while
Experiment **08** remains an external tool-use battery until the core frozen-
organ cortex condition succeeds. See each `research/NN-*.md` for verdict gates
and each experiment README for its build-out.
