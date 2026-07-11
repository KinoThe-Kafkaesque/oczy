# Experiment: Conversation World Model as the Cortex Modulator (RL Phase 0)

Research proposal: ../../research/07-conversation-world-model-rl.md

## Status

- **Implementation:** `src/oczy/experiments/conversation_world_model.py` — implemented and tested.
- **Campaign 0d48130 (2026-07-11):** **POSITIVE** (marker-free uptake) — `marker_free_uptake_gap=1.0`, `accept_pred_auc_string=1.0`, `accept_pred_auc_hidden=0.8125`. **NULL** (critic AUC improvement) — `critic_auc_delta=0.0`. Evidence: [`campaign log`](../../experiments_logs/2026-07-11_campaign_0d48130.md)


## Objective

Does a self-supervised acceptance predictor reading the LM hidden (a) predict corrections *before the correction text exists* better than the lexical `_looks_like_correction` stop-gap, and (b) when wired as the cortex `correction_signal`, recover uptake on marker-free corrections that the lexical gate misses — without over-firing on non-correction turns?

## Setup

- **Model / driver:** matched mock-vs-real. Real = `LlamaCVecDriver` on `LFM2.5-1.2B-Instruct-Q4_K_M.gguf`, `embedding=True`, `n_ctx=4096`, `n_threads=4` (so `peek_embedding` works; note the existing `run_curriculum.py` real-driver path uses the tighter `n_ctx=128`). Mock = the `_MockDriver` shared by the stressors (`n_embd=16, n_layers=2`, deterministic hash embedding) — used only as a no-semantics control where AUC must collapse to chance.
- **Cortex:** `KVCortexConfig(d_cortex=4)` (matches the real-driver curriculum config in `organism_curriculum/run_curriculum.py:38-43`); `articulate_scale=0.001`.
- **Predictor (the world model):** the existing `WorldModelCritic`, constructed as in `cortex_agent.py:219-226` (`use_hidden=True, use_value_head=True, mlp_hidden_units=16, value_learning_rate=0.05`). Acceptance head = `predict_acceptance(...).correction_likelihood`.
- **Scaffolds to reuse (extend, don't reinvent):**
  - `src/oczy/experiments/cortex_agent.py` — `perceive()` already takes an explicit `correction_signal` (`:331`); the modulator swap is the single variable.
  - `src/oczy/experiments/organism_curriculum/` — `dataset.py` (44 episodes, 6 stages, probe categories; `Episode.corrected_label`, `Probe.expected`), `scoring.py` (`matches`/`probe_matches`/`categorize_results` → outcome labels; imported in `run_curriculum.py:26`), `run_curriculum.py` (battery driver).
  - `world-model-critic/src/world_model_critic/critic.py` — predictor + TD value head (no change to its math).
  - `src/oczy/experiments/multi_fact_stressor.py` — `--paraphrase` / `Correction:` fact for the marker-strip condition.
- **Label definition (no markers used):** for each turn, `actual_outcome = corrected` iff the agent's answer fails `scoring.probe_matches` against `Episode.corrected_label`/`Probe.expected`; `accepted` otherwise. Initial episode answers → corrected (label 1); probe answers → mixed.

## Conditions / ablation matrix

Each pair changes exactly one variable.

| ID | Modulator source | Critic input | Corrections | Driver | Isolates |
|---|---|---|---|---|---|
| A0 | lexical `_looks_like_correction` | n/a (baseline gate) | marker-present | real | current behavior |
| A1 | world-model pred-error | hidden MLP | marker-present | real | modulator source (vs A0) |
| B0 | world-model pred-error | **string-only** (`use_hidden=False`) | marker-present | real | does the LM hidden add AUC |
| B1 | world-model pred-error | hidden MLP | marker-present | **mock** | semantic content needed (AUC→chance) |
| C0 | lexical | n/a | **marker-stripped** | real | lexical blindness (headline) |
| C1 | world-model pred-error | hidden MLP | **marker-stripped** | real | recovery vs C0 (headline) |

Main hypothesis pair: **A0 vs A1** (AUC + modulator quality) and **C0 vs C1** (marker-free uptake gap). Hidden-value pair: **A1 vs B0**. Semantic control: **A1 vs B1**.

## Procedure

1. **Corpus build.** Run all 6 curriculum stages through a fresh `CortexAgent` (real driver). For every turn (initial + each probe) log `(stage, episode_id, request, proposed_answer, lm_hidden, actual_outcome)`. Outcome from `scoring.probe_matches`, never from markers.
2. **Phase-0 pretrain.** Leave-one-episode-out: hold out one episode's turns, fit `WorldModelCritic` (teacher-forced `predict_acceptance` updates via `record_outcome`) on the rest, predict the held-out turns. Repeat over all 44 episodes; pool predictions.
3. **AUC.** Compute `accept_pred_auc` (rank-based) over the probe-balanced subset for A1 (hidden), B0 (string-only), B1 (mock). Lexical reference AUC = `_looks_like_correction(request)` as the score (expected ≈ 0.5 pre-correction).
4. **Modulator wiring.** Re-run the curriculum twice with identical seeds: arm A0 (lexical `correction_signal`) and arm A1 (`correction_signal = clip(|predicted_accept − realized_outcome|, 0, 1)` passed explicitly into `perceive()`). Record per-turn modulator value and `modulator_outcome_corr`.
5. **Marker-strip stressor.** Build C0/C1 by stripping `_CORRECTION_MARKERS` tokens from each correction utterance (regex remove of the marker phrases). Re-run uptake measurement: after the (stripped) correction, score the matching transfer probe; uptake = probe pass-rate.
6. **False-plasticity check.** On retention/scope probe turns (genuine non-corrections), record fraction where the modulator exceeds the plasticity-engaging threshold (`false_plasticity_rate`).
7. Re-run codebase_qa once per arm to confirm `code_qa_accuracy` is unchanged (regression guard only).

## Metrics

- **`accept_pred_auc`** — leave-one-episode-out AUC of `correction_likelihood` vs binary `actual_outcome`, probe-balanced subset. *New metric; replaces nothing (none existed). Headline acceptance metric is `marker_free_uptake_gap` (below): `accept_pred_auc` CAN saturate at 1.0 on a 44-episode leave-one-episode-out corpus if the predictor perfectly separates corrections from acceptances, so the >0.7 / >0.55 thresholds are read as "above / into the chance band" not as scaled headroom. The held-out AUC over a mixed-label set still cannot saturate trivially the way the all-ones legacy metrics did; treat the `rl_pipeline_design.md` Phase-1 >0.8 target as currently unmeasured.*
- **`marker_free_uptake_gap`** — uptake(C1) − uptake(C0), uptake = transfer-probe pass-rate after a marker-stripped correction. *Headline. Replaces the implicit assumption that markers == corrections. Non-saturating because it is a difference of two live arms.*
- **`false_plasticity_rate`** — fraction of non-correction turns with modulator above the plasticity-engaging threshold. Guards against the world-model simply firing always.
- **`modulator_outcome_corr`** — point-biserial correlation between the per-turn modulator value and the realized corrected outcome. *Replaces the binary lexical signal with a graded, measurable one.*
- **`code_qa_accuracy`** — reported only as a no-regression guard (expected to stay 1.0; **not** a success metric).

## Acceptance & kill criteria

- **Accept** if all hold: `accept_pred_auc(A1) ≥ 0.70` and `≥ accept_pred_auc(B0) + 0.05`; `marker_free_uptake_gap ≥ 0.40`; `false_plasticity_rate(A1) ≤ false_plasticity_rate(A0)`; `modulator_outcome_corr(A1) > 0` and `> A0`.
- **Kill** if any hold: `accept_pred_auc(A1) ≤ 0.55` or `≤ B0`; `marker_free_uptake_gap < 0.20`; `false_plasticity_rate(A1) > A0 + 0.10`. (Kill ⇒ the LM-hidden carries no answer-time acceptance signal at the final layer, escalating to project 03 layer-L extraction.)
- **Sanity gate:** B1 (mock) AUC must be ≈ 0.5 ± 0.07. If the mock control scores high, the AUC is leaking from string features / label order, not semantics — invalidate and fix before trusting A1.

## Controls

- **Modulator-source matched pair (A0/A1):** identical curriculum, seed, cortex, driver; only the `correction_signal` source differs.
- **Hidden-value matched pair (A1/B0):** identical except `use_hidden` True vs False — isolates the LM hidden's contribution over the 4 string features.
- **Semantic control (A1/B1):** identical except mock vs real driver — the no-semantics floor.
- **Marker matched pair (C0/C1):** identical stripped corrections; only modulator source differs — isolates lexical blindness.
- **Leave-one-episode-out** prevents the predictor from memorizing the held-out episode's tokens via the `_similar_correction_rate` record feature (`critic.py:442`).

## Expected failure modes

- **Final-layer hidden too coarse** → A1 AUC ≈ B0 (string-only) ⇒ kill, route to project 03.
- **Label imbalance** inflating AUC if computed over all-corrected initials → enforce probe-balanced subset; B1 sanity gate catches leakage.
- **Continuous modulator saturates plasticity** (`alpha_correction=5.0` then clip to [0,1]) → uptake numerically OK but `false_plasticity_rate` high; mitigate by scaling pred-error into the [0,1] band before passing to `observe()`.
- **Marker-strip changes meaning** (removes load-bearing words) → confounds C0/C1; restrict stripping to the exact `_CORRECTION_MARKERS` phrases and assert the `corrected_label` token survives.
- **Tiny corpus, wide CI** → AUC inconclusive; report bootstrap CI and treat overlapping CIs with B0 as a kill.

## Artifacts to add

- `src/oczy/experiments/conversation_world_model.py` — corpus builder + leave-one-episode-out trainer/evaluator + modulator-swap harness; emits `METRIC ...` lines for the autoresearch parser (`accept_pred_auc`, `marker_free_uptake_gap`, `false_plasticity_rate`, `modulator_outcome_corr`).
- `src/oczy/experiments/tests/test_conversation_world_model.py` — mock-driver unit tests: AUC≈0.5 sanity, modulator-swap wiring into `perceive()`, marker-strip preserves `corrected_label`.
- `experiments_logs/2026-06-28_conversation_world_model_rl.md` — run notes (matched-pair tables, AUC + CI).

Sketch reproduce command:

```
uv run python -m oczy.experiments.conversation_world_model \
  --driver real --loo --stages all \
  --arms A0,A1,B0,B1,C0,C1 --marker-strip-phrases default
```
