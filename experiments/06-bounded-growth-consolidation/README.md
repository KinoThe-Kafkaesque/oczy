# Experiment: Bounded-Growth Consolidation — Trained Encoder + Concept-Embedding Hypernetwork

Research proposal: ../../research/06-bounded-growth-consolidation.md

## Objective

Can replacing the random-projection `ExperienceAutoencoder` and the per-concept-row `IdentityHypernetwork` with a trained compact encoder and a trained concept-*embedding* hypernetwork cut the combined serialized footprint of those two organs by **≥10x** — and the whole-organism `memory_bytes_per_behavior_delta` by **≥2x** — *without* dropping any eval-suite behavior score below the current `OrganismAgent` (transfer 0.25 / scope 0.1667 / forgetting 1.0 / identity 1.0)?

## Setup

- **Driver:** none. Both target organs are pure NumPy (autoencoder.py, hypernet.py) and the eval-suite curriculum drives string-feature organs (`PlasticCortex` word-association etc.), so no LFM2.5 GGUF is loaded for the primary measurement. This is honest: the bloat is in organ serialization (`pickle.dumps`), not in the LM. An *optional* real-driver cross-check uses `organism_curriculum/run_curriculum.py --use-real-driver` (cross-link 05).
- **Reused scaffolds (real paths):**
  - `src/oczy/experiments/eval_suite.py` — `EvalSuite.run(agent)` produces `final_card.memory_bytes_per_behavior_delta`, `transfer_score`, `scope_score`, `forgetting_score`, `consolidation_score`, `identity_drift_score`, plus `raw_trace_size` / `consolidated_size` on the `EvalResult` (eval_suite.py:396,416-435).
  - `src/oczy/experiments/run_experiment.py` — the baseline registry (ZeroMemory/ContextOnly/FastOnly/HippocampusOnly/IdentityOnly/Organism/Null; run_experiment.py:135-143) for the 12 B/Δ floor and 68,772 B/Δ reference.
  - `src/oczy/experiments/organism.py` — `OrganismAgent`; note `config["experience_autoencoder"]` is passed positionally into the fixed `ExperienceAutoencoder` constructor and `config["identity_hypernetwork"]` is splatted as kwargs into the fixed `IdentityHypernetwork` constructor (organism.py:63-69), so a *replacement organ* is injected by overwriting `agent.experience_autoencoder` / `agent.identity_hypernetwork` after construction (or a small constructor extension), not via a config value; `_module_bytes` per-organ decomposition (organism.py:554-588).
  - `experience-autoencoder/src/experience_autoencoder/autoencoder.py` — encoder/decoder to replace; `decode()` field set (autoencoder.py:264-269) is the reconstruction target.
  - `identity-hypernetwork/src/identity_hypernetwork/hypernet.py` — hypernetwork to replace; `generate_adapters` / `update_identity` / `grow_vocab` surface.
  - `eval_extended.py:406-443` — reconstruction-fidelity scoring (`reconstruction_error` component, `1 - mean`) for held-out corrections.
- **Curriculum / corpus:** the `EvalSuite` curriculum (fixed `curriculum.seed`) for behavior + byte metrics; the `eval_extended.py` `CURRICULUM` (30 word-sense corrections + 12 trivia, per EVALUATION.md:3) split train/held-out for encoder training and reconstruction scoring. Stages 0/1/4 of `organism_curriculum` (which "absorb cleanly", NOTES.md:243-245) are the optional real-driver cross-check.
- **Latent sizing:** `LATENT_DIM ∈ {8,16,32}` for the trained encoder (default 32; autoencoder.py:24); hypernetwork `latent_dim ∈ {8,16,32}` (default 8; hypernet.py:59).

## Conditions / ablation matrix

Single variable per matched pair: *which organ implementation* is swapped. Same curriculum seed, same episodes, same `successful_lessons` denominator.

| ID | Encoder | Hypernetwork | Isolates |
|----|---------|--------------|----------|
| A0 | random projection, full `_A` pickled (current) | seed vocab + auto-`grow_vocab` (current) | baseline (NOTES.md:256 = 68,772 B/Δ) |
| A0b | random projection, **persist seed + Hebbian deltas only** (regenerate `_A` on load) | current | serialization-only byte win (no training) |
| A1 | **trained compact encoder** | current | encoder training, in isolation |
| A2 | current | **trained concept-embedding** (no per-concept `W` rows) | hypernetwork bounded-vocab, in isolation |
| A3 | trained compact encoder | trained concept-embedding | full compact (composition) |
| REF-lo | — (FastOnlyAgent) | — | absolute byte floor (NOTES.md:253 = 12 B/Δ) |

Plus a sweep on A3 over `latent_dim ∈ {8,16,32}` to expose the compression-penalty vs behavior tradeoff.

## Procedure

1. **Per-organ decomposition (step 0).** Instantiate A0 `OrganismAgent`, run `EvalSuite.run`, and record `_module_bytes` for each of the 6 organs (organism.py:554-588). Confirm what fraction of `consolidated_size` the autoencoder + hypernetwork own. (Gate on kill-criterion d before investing in training.)
2. **A0b verification.** Add a `seed_regenerable` serialization mode to the autoencoder; confirm A0b's eval-suite behavior is byte-identical to A0 (proves `_A` is never trained in the default flow, organism.py:415-416) and record its `serialized_bytes`. (In the default flow there are zero Hebbian deltas, so A0b persists only the seed plus the grown token-vocab dict.)
3. **Train the compact encoder offline** on the train split, minimizing the thesis-9 composite loss (reconstruction of `decode()` fields + compression penalty + anti-overgeneralization on held-out + replay consistency). Persist only the trained weights.
4. **Train the concept-embedding hypernetwork**: replace the per-concept `W` rows with a fixed `(concept_dim, 4·latent_dim)` shared embedding; train so `generate_adapters(z)` reproduces the curriculum's correct-concept ranking.
5. **Build A1, A2, A3** by constructing the `OrganismAgent` and then overwriting `agent.experience_autoencoder` / `agent.identity_hypernetwork` with the trained organs (since organism.py:67-69 only passes those config keys as constructor arguments to the fixed organ classes, not as prebuilt instances; optionally extend the constructor to accept a prebuilt organ). Run `EvalSuite.run` for each; record the full `final_card` + per-organ bytes.
6. **Growth instrumentation:** during each run, sample `agent.memory_bytes()` after every correction; fit an OLS slope (B/correction) over corrections 4..N (post-warmup).
7. **Reconstruction scoring:** run the eval_extended-style reconstruction (eval_extended.py:428-437) on the **held-out** corrections for A0/A0b vs A1/A3.
8. **Optional real-driver cross-check:** `run_curriculum.py --use-real-driver` on stages 0/1/4 to confirm the byte/delta ordering holds with real LFM2.5 embeddings (run_curriculum.py tracks `memory_bytes_before/after` per stage).

## Metrics

- **M1 — combined controllable footprint (PRIMARY, new framing).** `serialized_bytes(experience_autoencoder) + serialized_bytes(identity_hypernetwork)` via `_module_bytes` (organism.py:554-588). Report the ratio A3/A0. Does not saturate: A0 ≈ 229 KB autoencoder alone.
- **M2 — whole-organism byte/delta (SECONDARY).** `final_card["memory_bytes_per_behavior_delta"]` = `consolidated_size / max(1, successful_lessons)` (eval_suite.py:396). Extends the existing metric; spans 12→68,772 today so it cannot saturate at 1.0.
- **M3 — reconstruction fidelity (QUALITY).** Held-out `1 - mean(reconstruction_error)` (eval_extended.py:432-437). Replaces the EVALUATION.md 0.203 aggregate's reconstruction half; baseline-relative so it cannot saturate trivially.
- **M4 — behavior guards.** `transfer_score`, `scope_score`, `forgetting_score`, `identity_drift_score` from `final_card` (eval_suite.py:418-423). `forgetting`/`identity` are saturated-at-1.0 floors (regression guards), not discriminating axes. Report the `successful_lessons` denominator separately so M2 is not confounded by behavior change.
- **M5 — marginal growth slope (BOUNDEDNESS, new).** OLS slope of `memory_bytes()` vs correction index, post-warmup. Continuous regression; no ceiling.

## Acceptance & kill criteria

**Accept** if A3 (best `latent_dim`) achieves *all*:
- M1 ratio `≤ 0.10` (≥10x combined-footprint reduction vs A0).
- M2 `≤ 34,386` (≥2x vs 68,772).
- M3 `> 0.203` on held-out.
- M4: `transfer ≥ 0.25`, `scope ≥ 0.1667`, `forgetting = 1.0`, `identity = 1.0`.
- M5 `≤ 100 B/correction`.

**Kill** if any: best M1 ratio `> 0.5` (<2x); any compressed condition drops `forgetting` or `identity` below 1.0; M3 `≤ 0.203`; or step-0 decomposition shows the two organs own `< 30%` of `consolidated_size` (rescope trace compression to 05-metabolism-loop-closure).

## Controls

- **Matched pairs (single variable):** A0↔A0b isolates *serialization strategy* (bytes only, behavior must be identical); A0b↔A1 isolates *encoder training*; A0↔A2 isolates *hypernetwork vocabulary*; (A1,A2)↔A3 isolates *composition*.
- **REF-lo (FastOnlyAgent, 12 B/Δ; NOTES.md:253):** absolute lower bound; the A3→REF-lo gap quantifies the irreducible cost of carrying a compact identity + hypernetwork over pure fast-weights.
- **Mock-vs-real:** primary measurement is driver-free NumPy; optional `--use-real-driver` cross-check (organism_curriculum) confirms the ordering survives real embeddings.
- **Seed control:** fixed `curriculum.seed` and fixed organ `seed` across all conditions so `successful_lessons` (the M2 denominator) is held constant.

## Expected failure modes

- **Hippocampus dominates `consolidated_size`** → M2 caps well under 2x even with M1 ≥10x; surfaced by step-0 decomposition (kill-d), handoff to 05.
- **Over-compression at `latent_dim=8`** collapses `scope_score` (already 0.1667; NOTES.md:256) — the anti-overgeneralization term fails, or the documented two-senses-per-token cortex limit (Stage 2 100% fail; NOTES.md:243) bounds scope regardless of encoder.
- **Concept-embedding underfits** the curriculum's distinct concepts → transfer regresses; note the fixed-vocab "complete failure" was a pre-`grow_vocab` artifact (current re-validated score 0.006, NOTES.md:126), so the embedding must beat the *current* auto-grown hypernetwork on transfer, not just the 0.001 number.
- **A0b is not behavior-identical to A0** → the "_A never trained" assumption (organism.py:415-416) is wrong; re-derive the seed-regenerable claim before trusting M1.
- **Trained encoder overfits train split** → held-out M3 collapses despite high train reconstruction (the replay-consistency term is doing nothing).

## Artifacts to add

- `src/oczy/experiments/bounded_growth/train_encoder.py` — offline trainer for the compact `ExperienceEncoder` (thesis-9 composite loss); saves compact weights + a `seed_regenerable` serialization mode.
- `src/oczy/experiments/bounded_growth/concept_embedding_hypernet.py` — `IdentityHypernetwork` variant with a fixed-size shared concept embedding replacing per-concept `W` rows.
- `src/oczy/experiments/bounded_growth/bounded_growth_eval.py` — builds A0/A0b/A1/A2/A3 + REF-lo `OrganismAgent` variants (overwriting the autoencoder/hypernetwork attributes post-construction), runs `EvalSuite.run`, does per-organ `_module_bytes` decomposition, fits the M5 growth slope, and prints `METRIC-`/`ASI-` lines for the autoresearch harness (matching the multi_fact_stressor convention).
- `reports/bounded_growth/run.json` — per-condition `final_card` + per-organ bytes + slope.

Sketch reproduce command:

```
uv run python -m oczy.experiments.bounded_growth.bounded_growth_eval \
    --conditions A0,A0b,A1,A2,A3 --latent-dims 8,16,32 --seed 0 \
    --report reports/bounded_growth/run.json
```
