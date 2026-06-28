# Experiment: Compounding Cold-State Drift Drives the Answer

Research proposal: ../../research/05-metabolism-loop-closure.md

## Objective

When the same concept is corrected K times through `CortexAgent` (each followed by `consolidate()`), does `cortex.cold_state` drift **compound** (accumulate, not overwrite), and does that accumulated drift — with every label/prefix/logit-bias surface disabled — drive a **continuous, K-increasing domain shift** in the LM's answer?

## Setup

- **Driver:** matched pair. (a) Real `LlamaCVecDriver` on `LiquidAI/LFM2.5-1.2B-Instruct-Q4_K_M.gguf`, following the real-driver loader pattern in `multi_fact_stressor.py::_load_real_driver` / `smoke_consolidation_uptake_compare.py`. (`n_ctx=512` / `n_threads=4` are the `CVecDriverConfig` *defaults*, but the cited loaders override them — `_load_real_driver` uses `n_ctx=4096`/`n_threads=4`, the smoke scaffold uses `n_ctx=128`/`n_threads=12`; `embedding=True` is required for `peek_embedding`. Pin one `n_ctx` across all conditions.) (b) `_MockDriver` from `multi_fact_stressor.py` (constructed `_MockDriver(n_embd=16)`; `n_layers=2` is a fixed attribute, not a kwarg) as the semantic-null control.
- **Agent:** `CortexAgent` (`src/oczy/experiments/cortex_agent.py`) via `CortexAgentConfig`. `cortex=KVCortexConfig(d_cortex=8, steering_mode="proj_random")`, `auto_consolidate=False` (overriding the `True` default; we call `consolidate()` explicitly to control K), `articulate_scale=0.03` (the known-good SVD-init proj_c band, which is the configuration this experiment uses because step 3 calls `init_proj_c_from_svd`; sweep {0.001 raw_hidden, 0.03 svd} as a scale control). `use_logit_bias=False`, no articulation prefix in drift-only conditions.
- **Reused scaffolds (extend, do not reinvent):**
  - `smoke_consolidation_uptake_compare.py` — the correction loop (`agent.turn(correction, correction_signal=1.0, max_tokens=4, temperature=0.0)`, looped 8× to collect ≥d_cortex hiddens), `init_proj_c_from_svd(np.vstack(hiddens))`, hard `consolidate(strength=agent.config.cortex.max_consolidation_strength)`, `_contains` and the `domain_words` list.
  - `cortex_agent.py` — `turn()`, `consolidate()` (returns `cold_drift`), `cortex.status()` (`cold_norm`, `warm_cold_drift`).
  - `multi_fact_stressor.py` — real/mock driver loaders, METRIC-/ASI- print lines for the autoresearch harness.
  - `world-model-critic/src/world_model_critic/critic.py` — `record_outcome`, `predict_acceptance` (for the critic-conversion arm).
  - `neural-hippocampus/src/neural_hippocampus/{hippocampus,core}.py` — `store(hidden=...)`, `consolidate()`, retrieval (`core.py:_embed`).
- **Concept/probe:** reuse the proven probe `"'Profile' here means business _______."`, target domain word set `[commercial, economic, business, strategy, market, vertical]`, correction `"No, 'profile' here means business vertical, not user profile."` (from `smoke_consolidation_uptake_compare.py`). Add a 2nd held-out concept for the scope/cross-talk control.

## Conditions / ablation matrix

Single-variable matched pairs (one axis varies per row vs the `compound-real` anchor):

| Condition | Varied axis | Driver | Replay path | Label/prefix | Critic mode |
|---|---|---|---|---|---|
| `compound-real` (anchor) | — | real | tensor-keyed, on | none | string |
| `overwrite-null` | replay branch | real | off (threshold=∞) | none | string |
| `hash-replay` | replay addressing | real | hash-keyed | none | string |
| `drift-only` | label surface | real | on | none | string |
| `label-only` | label surface | real | n/a (zero cold) | prefix | string |
| `zero-baseline` | drift present | real | n/a (zero cold) | none | string |
| `compound-mock` | semantics | mock | on | none | string |
| `scale-001` | articulate_scale | real | on | none | string |
| `critic-string` | critic input | real | on | none | string features |
| `critic-drift` | critic input | real | on | none | `warm_cold_drift` |

## Procedure

1. Build `compound-real` agent; record `cortex.status()` baseline (`cold_norm`, `warm_cold_drift`).
2. Loop k=1..K (K=20): call `agent.turn(correction, correction_signal=1.0, max_tokens=4)`; collect `agent._last_hidden`; call `agent.consolidate(strength=agent.config.cortex.max_consolidation_strength)`; append the returned `cold_drift`, the post `cold_norm`, `warm_cold_drift`, and `cold_state.copy()`.
3. After warm-up of ≥`d_cortex` correction hiddens (with `d_cortex=8`, that means ≥8 corrections; `init_proj_c_from_svd` raises `ValueError` if `N < d_cortex`), call `cortex.init_proj_c_from_svd(np.vstack(hiddens))` (matches the smoke scaffold), then continue corrections so the SVD direction is in the persisted projector. Because steering at `articulate_scale=0.03` only takes effect once proj_c is SVD-initialised, treat probe points before this warm-up as the structurally-flat baseline.
4. At k ∈ {1,2,5,10,20} run the held-out probe with steering on, score the **domain-shift** metric (below). For `drift-only` ensure no prefix/bias is set; for `label-only` use a fresh agent with zero cold (`reset_warm_to_zeros` on a freshly-booted cortex) and set the prefix; for `zero-baseline` neither.
5. Repeat steps 1–4 for every matrix row, holding all non-varied config identical (`dataclasses.replace`).
6. Critic arm: replay the same K corrections plus an equal number of *acceptance* turns through `critic-string` and `critic-drift`; for `critic-drift`, feed `cortex.status()["warm_cold_drift"]` as the `record_outcome` error target; collect `predict_acceptance` correction-likelihoods for an AUC.
7. Emit METRIC-/ASI- lines and dump per-step arrays to `reports/metabolism_loop/<condition>.json`.

## Metrics

- **`cold_norm_slope`** — OLS slope of `cold_norm` vs k over K steps; bootstrap CI (resample k-pairs). *Replaces* the binary `consolidated: True` flag; continuous, cannot saturate. Reference: 2026-06-25 grew 0.34→3.6 / 20 cycles.
- **`compounding_index`** — `‖Σ_k Δcold_k‖ / Σ_k ‖Δcold_k‖` ∈ (0,1], where `Δcold_k = cold_after − cold_before` per step (the per-step `‖Δcold_k‖` is exactly the `cold_drift` already returned by `CortexAgent.consolidate()`; the vector deltas come from the `cold_state.copy()` snapshots in step 2). 1.0 = perfectly additive/compounding; →0 = cancellation/overwrite.
- **`domain_shift(k)`** — PROPOSED continuous read-out: mean next-token logit over `domain_words` token ids at the probe's blank position, read via the driver's logit machinery (the same `get_logits()` surface used by `logit_bias_generate` and the `contrastive_cvec_discovery` rank/logit measurement), minus the `zero-baseline` value. This is a *new* metric (the smoke scaffold currently scores domain words by substring `_contains`, which is binary); it *replaces* `domain_co_recall` (saturated 1/1) and `exact co_recall` (floored 0/0): we require it to *grow with k*, which a saturated binary cannot express.
- **`drift_label_dissociation`** — `domain_shift` of `drift-only` (no label) vs `label-only` (no drift); both > 0 and uncorrelated proves drift is an independent carrier.
- **`spearman_rho(domain_shift, k)`** — monotonicity test for H2.
- **`correction_pred_auc`** — ROC-AUC of critic correction-likelihood (`predict_acceptance` → `correction_likelihood`) vs actual correction labels on the held-out mixed sequence, for `critic-string` and `critic-drift`; report the matched-pair delta (RL design P1 gate is AUC>0.8, but the *delta* is the discriminator).
- **`replay_fired_fraction`** — fraction of `consolidate()` calls where `len(replays) ≥ consolidate_replay_threshold` (=3) (proves the additive branch engaged); compared tensor-keyed vs hash-keyed.

## Acceptance & kill criteria

- **Accept H1** if `compound-real` `cold_norm_slope` CI excludes 0 AND `compounding_index ≥ 0.6`, while `overwrite-null` slope ≤ 0.25× anchor slope (or CI includes 0).
- **Accept H2** if `drift-only` `spearman_rho ≥ 0.5` (p<0.05) and its K=20 `domain_shift` bootstrap CI excludes 0, while `zero-baseline` CI includes 0.
- **Accept critic conversion** if `critic-drift` AUC − `critic-string` AUC > 0 with CI excluding 0.
- **Accept tensor replay bank** if `replay_fired_fraction(tensor) > replay_fired_fraction(hash)` and `hash-replay` `compounding_index < 0.6`.
- **Kill:** `domain_shift` flat after k=1 (no compounding behavior) → metabolize thesis fails, report honestly. `compounding_index < 0.3` everywhere → overwrite confirmed. `drift-only` CI includes 0 → result gated on `03-layer-l-hidden-extraction` (final-layer signal too weak). Pre-registered: exact-token recall is explicitly NOT an acceptance metric (falsified 2026-06-25).

## Controls

- **Mock vs real** (`compound-mock` vs `compound-real`): isolates semantics; mock `domain_shift` must be ≈0.
- **Replay on/off** (`overwrite-null` vs anchor): isolates the additive consolidate() branch as the compounding mechanism.
- **Hash vs tensor keying** (`hash-replay` vs anchor): isolates the replay-bank addressing as load-bearing for `replay_fired_fraction`.
- **Drift vs label** (`drift-only`/`label-only`/`zero-baseline`): isolates what carries the behavior.
- **Scale** (`scale-001` vs anchor): confirms domain-shift is not a scale artifact (raw_hidden 0.001 vs SVD-init 0.03).
- **Critic input** (`critic-string` vs `critic-drift`): isolates the Goal-3 critic conversion; same episodes.

## Expected failure modes

- Norm grows 10× but `domain_shift` is flat — the 2026-06-25 off-manifold trap; caught because C2 measures behavior, not norm.
- Slow-EMA-only path saturates (`overwrite-null` slope ≈0) — expected, and is the evidence that the additive replay branch is required. (Caveat: warm itself can grow with repeated corrections, so the slow nudge can still produce some growth toward a rising warm; `compounding_index` and the replay-on/off contrast disentangle this.)
- Real-driver `domain_shift` noisy at 1.2B — mitigate with multiple probe paraphrases + bootstrap; if still ≤ noise, escalate to layer-L (project 03).
- Hash-keyed retrieval never reaches `≥3` clustered replays, so replay branch never fires (`replay_fired_fraction≈0`) — expected null that motivates tensor keying.
- `init_proj_c_from_svd` needs N≥d_cortex hiddens; with d_cortex=8 ensure ≥8 corrections before SVD-init or the call raises `ValueError` / the projector stays random.

## Artifacts to add

- `src/oczy/experiments/metabolism_loop.py` — driver harness: runs the K-correction loop, logs `cold_norm`/`warm_cold_drift`/`cold_drift` trajectories, runs domain-shift probes across conditions, computes `compounding_index`, `cold_norm_slope`, `spearman_rho`, `correction_pred_auc`; emits METRIC-/ASI- lines.
- `src/oczy/experiments/metabolism_critic_drift.py` — thin `WorldModelCritic` subclass/wrapper that accepts `warm_cold_drift` as the `record_outcome` error target (Goal 3 row "WorldModelCritic → `cortex.warm_cold_drift`").
- `src/oczy/experiments/tensor_replay_bank.py` — optional retrieval shim over `NeuralHippocampus` that keys retrieval/clustering on the stored `hidden` (cosine, d_embd) instead of the sha256 text hash (`core.py:_embed`).
- `experiments_logs/2026-06-28_metabolism_loop_closure.md` — results table + plots; update `src/oczy/experiments/logs/SUMMARY.md`.

Sketch reproduce command:

```
uv run python -m oczy.experiments.metabolism_loop \
  --real-driver --concept profile --corrections 20 \
  --conditions compound-real,overwrite-null,hash-replay,drift-only,label-only,zero-baseline \
  --articulate-scale 0.03 --d-cortex 8 --out reports/metabolism_loop
```
