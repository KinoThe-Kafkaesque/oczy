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

Test status: `pytest: 186 passed` (170 fast + 16 slow/model), 6 deprecation\n+warnings only. `ruff check` clean on changed files.

Remaining blocks (not in the optimization list):
- Direct reserved KV-slot injection still blocked by `llama-cpp-python` C API surface.
- Exact-token uptake via cvec alone remains blocked; `ReservedPosition` prefix is the\n+  practical exact-recall surface.
