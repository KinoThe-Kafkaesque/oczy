# Experiment Summary — 2026-06-25

Seed: 0 | Sense matching: enabled | Consolidation: enabled

| Agent | Uptake | Transfer | Scope | Forget | Consol | Identity | Mem/Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ZeroMemoryAgent | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 69.0 |
| ContextOnlyAgent | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 684.75 |
| FastOnlyAgent | 0.6667 | 0.1667 | 0.1667 | 1.0000 | 1.0000 | 1.0000 | 12.0 |
| HippocampusOnlyAgent | 0.6667 | 0.1667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.25 |
| IdentityOnlyAgent | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 493.0 |
| OrganismAgent | 0.6667 | 0.2500 | 0.1667 | 1.0000 | 1.0000 | 1.0000 | 68636.5 |

## Key observations

- FastOnlyAgent remains the most memory-efficient learner (12 bytes/Δ).
- OrganismAgent achieves the highest transfer (0.25) but pays a very large
  memory cost, so its composite behavior-delta-per-byte score is still low.
- HippocampusOnly and IdentityOnly baselines continue to struggle with transfer
  in the simple word-association domain, showing the standalone organs are not
  yet compressing lessons efficiently.
- The central bottleneck remains consolidation: the experience autoencoder and
  identity hypernetwork store large serialized objects rather than compact
  adapters.
- Recent autoresearch work on the codebase-QA benchmark reached
  `code_qa_accuracy=1.0` and `cortex_agent_recall_accuracy=1.0`.
- Consolidation-uptake probes showed boot-persistent *domain* shift via cvec
  steering, but exact-token uptake failed until a soft-prompt prefix was used.
  See `experiments_logs/2026-06-25_prefix_steering_poc.md` for the prefix
  steering proof of concept.

## Session delta (2026-06-25 continuation)

Commits since previous summary:
1. `8ee8d8e` — Make hippocampus replay tensor-native: hidden vectors stored with traces,\n+   `consolidate()` uses mean-cluster hidden replays, `cold_drift=0.324` in manual probe.
2. `87a7779` — Optimize LM boundary and status serialization:\n+   - embedding cache in `LlamaCVecDriver.peek_embedding`,\n+   - shared SVD projector / uniform cvec path in `KVCortex`,\n+   - optional `serialized_bytes` in organ `status()` methods.
3. `52f257f` — First-class `ReservedPosition` abstraction replaces literal\n+   `articulation_prefix`; LM perception parser hardened for short ambiguous tokens;\n+   ruff/pyright/pytest markers added to `pyproject.toml`.
4. `a5468b6` — Complete remaining review items:\n+   - bound linear-growth organs (`WorldModelCritic`, `IdentityHypernetwork`,\n+     `SurpriseGatedMemory`) with configurable caps and decay,\n+   - driver profiles + `OCZY_*` env-aware config for `CVecDriverConfig` and\n+     `LanguageAdapterConfig`,\n+   - versioned non-pickle `KVCortex` persistence via `manifest.json` + `arrays.npz`.
5. `7450067` — Tidy tooling config and fix ruff warnings.
6. `a9cca21` — Update `GOALS.md` to mark reserved-position API implemented.
7. `3985f04` — Wire ReservedPosition selection from the KnowledgeStore into
   `CortexAgent.articulate()`. Added `KnowledgeStore.get_reserved_position()` and
   tagged five facts with `reserved_token` metadata. `CortexAgent.articulate()` now
   applies the reserved prefix when recalling facts and skips cvec steering to avoid
   cvec+prefix interference. Benchmark: `code_qa_accuracy=1.0` (run #49).
8. `040ed56` — Add optional tensor-input MLP to `WorldModelCritic` and wire
   `CortexAgent.metabolize()` to pass `self._last_hidden` to predict/record calls.
   The string-only logistic path is preserved; the MLP path is gated by
   `use_hidden=True` and lazy-initializes on first hidden vector. Benchmark
   unchanged: `code_qa_accuracy=1.0` (run #50).
9. `b951011` — Add a signed SGD replay train step on `KVCortex.proj_hidden` and
   wire `CortexAgent.consolidate()` to call it per hippocampal summary.
   Correction summaries reinforce the response direction, neutral summaries
   suppress it.  Gated off by default via `KVCortexConfig.replay_sgd_step`.
   Fast tests pass; benchmark unchanged: `code_qa_accuracy=1.0` (run #51).
10. `2f7d116` — Wire IdentityHypernetwork state adapters into `KVCortex`
    articulation bias. `IdentityHypernetwork` now learns per-concept
    `state_adapters` via EMA during `update_identity` and emits a real
    `d_cortex`-dimensional adapter delta. `CortexAgent.metabolize()` applies
    this delta through `KVCortex.set_state_bias`, which is added to
    `warm_state` during cvec emission. Benchmark unchanged:
    `code_qa_accuracy=1.0` (run #54).

Test status: `pytest: 205 passed` fast (reserve-position + tensor-critic + replay-SGD +
identity-adapter unit tests pass; full slow/model suite not rerun). `ruff check` clean
on changed files.

Remaining blocks:
- Direct reserved KV-slot injection still blocked by `llama-cpp-python` C API surface.
- Exact-token uptake via cvec alone remains blocked; `ReservedPosition` prefix is the
  practical exact-recall surface, and it can now be selected automatically by the
  knowledge store.
- WorldModelCritic now has a tensor-input MLP path but it is not yet the default.
- Hippocampal replay now has a differentiable SGD path on `proj_hidden`, gated by
  `replay_sgd_step` and defaulting to off.
- IdentityHypernetwork now emits real `d_cortex`-dimensional adapter deltas that are
  applied at articulation time, but the concept→latent mapping is still partially
  hand-seeded and the effect on downstream behavior has not yet been measured.
