# 07 — Conversation World Model (RL Phase 0)

*An agent should predict the correction before it arrives — then let that prediction error, not a keyword, decide what to learn.*

**Status:** ACCEPTED-PARTIAL (2026-07-11) | **Thesis anchor:** experiments.txt §6 (world-model-first: predict OUTCOMES, correction = prediction error); supports §4 (learned plasticity) and §10 (mistake immune system) | **Goal anchor:** GOALS.md Goal 3 (upgrade organs from string features to tensor consumers so the metabolism loop closes); relates to Goal 2 (layer-L peek feeds richer hidden) | **Depends on / relates to:** 01-correction-to-competence-benchmark, 05-metabolism-loop-closure (cross-link), 03-layer-l-hidden-extraction (richer input signal)

> **Outcome (2026-07-11):** ACCEPTED-PARTIAL. Campaign 0d48130 (colab, commit `537260c`): **POSITIVE** for marker-free uptake — `marker_free_uptake_gap=1.0` (≥ 0.40 threshold met), `accept_pred_auc_string=1.0`, `accept_pred_auc_hidden=0.8125` (≥ 0.70 threshold met). **NULL** for critic AUC improvement — `critic_auc_delta=0.0` (the hidden-feature critic does not beat the string-feature critic). H1 (predictive) accepted; H2 (behavioral) partially accepted: the world-model modulator recovers marker-free uptake but does not improve correction prediction over string features. Single-run, no cross-seed variance. Evidence: `../experiments_logs/2026-07-11_campaign_0d48130.md`.

## Problem

The cortex decides *how much to learn* from a turn via a single scalar `correction_signal` that gates plasticity in `KVCortex.observe()` (`plasticity = alpha_warm*(1-c) + alpha_correction*c`). Today that scalar is produced by a lexical stop-gap: `perceive()` sets `correction_signal = 1.0 if _looks_like_correction(utterance) else 0.0` (`src/oczy/experiments/cortex_agent.py:347-348`), where `_looks_like_correction` (`cortex_agent.py:68-71`) is a substring match against a hand-coded `_CORRECTION_MARKERS` tuple ("no, ", "wrong, ", "correction:", "expected:", "i meant", "actually,", "rather than", …; `cortex_agent.py:53-65`).

This is a keyword detector wearing the mask of a world model. It fails in two documented ways:

- **It is blind to marker-free corrections.** A correction phrased without a marker ("project beta uses rook", a paraphrase, an implicit re-grounding) reads as `correction_signal=0`, so the cortex applies only the `alpha_warm` (0.02) baseline plasticity and the lesson is not absorbed. The salience-ablation run #82 already hit this edge case from the other side — "correction-marker salience embedded all chunks in the synthetic needle turn … because the synthetic needle contains no marker" (2026-06-26 salience log) — confirming marker presence and semantic correction are not the same event.
- **The actual world model is built but never used as the modulator.** `CortexAgent` constructs a `WorldModelCritic` with `use_hidden=True, use_value_head=True, mlp_hidden_units=16, value_learning_rate=0.05` (`cortex_agent.py:219-226`). It has a real acceptance head (`predict_acceptance` → `accepted_prob, correction_likelihood, key_uncertainty`, reading a 16-unit MLP over `[4 string features ; lm_hidden]`, `critic.py:171-199, 425-440`) and a TD(0) value head (`reward=-1/+1, gamma=0.95`, `critic.py:319-327`). But its prediction is consumed only as a digest scalar; it never replaces `correction_signal`, and per the RL/policy SUMMARY the value head "is trained with TD on every metabolize() … but [has] not been validated in a real correction/uptake loop". `rl_pipeline_design.md` lists Phase 0 (predictive foundation) and Phase 1 (critic acceptance AUC > 0.8) as design-only, and the RL brief's own open question states "No held-out perplexity / Phase 0 predictive-foundation experiment has been run."

So the predictive substrate thesis §6 demands ("Then correction becomes prediction error") exists in code but is wired as a passenger, while a keyword heuristic drives the steering wheel.

## Hypothesis

1. **(Predictive)** A self-supervised acceptance predictor reading the LM hidden (`WorldModelCritic` with `use_hidden=True`) can predict, *at answer time and before any correction text exists*, whether a turn will be corrected, with held-out (leave-one-episode-out) AUC strictly above both chance (0.5) and a string-features-only critic (`use_hidden=False`). The lexical `_looks_like_correction` detector has AUC ≈ 0.5 on this pre-correction task by construction (the marker is not yet present).
2. **(Behavioral)** Replacing the lexical `correction_signal` with the world-model's realized prediction error as the cortex modulator recovers correction uptake on **marker-free** corrections that the lexical detector misses, with a marker-free uptake gap Δ ≥ 0.4 (world-model − lexical), **without** raising false plasticity on non-correction (retention/scope) turns.

Either claim is independently falsifiable.

## Why now / what unblocks it

- The predictor already exists and is already fed the right tensor: `metabolize()` calls `predict_acceptance(..., lm_hidden=hidden_for_critic)` and `record_outcome(..., lm_hidden=...)` every turn (`cortex_agent.py:463-478`), so a trace corpus of `(hidden, proposed_answer, actual_outcome)` tuples is already flowing — it is simply discarded as a digest scalar instead of being trained and read back.
- `perceive()` already accepts an explicit `correction_signal` argument (`cortex_agent.py:331`), so swapping the modulator source is a **clean single-variable** change: pass the world-model prediction error instead of letting the lexical default fire. No cortex surgery.
- The labels are free: the organism curriculum (`organism_curriculum/`, 44 episodes across 6 stages: 8/8/8/4/10/6) plus its probe batteries already produce accept/correct outcomes via the curriculum's `organism_curriculum/scoring.py` (`matches`/`probe_matches`/`categorize_results`, imported in `run_curriculum.py:26`; probe categories transfer|scope|forgetting|retention). Initial episode answers are always corrected (label=1); post-learning probes are accepted-or-corrected (mixed labels). This is the "interaction traces" corpus Phase 0 asks for, minus the unimplemented recurrent-cortex perplexity objective.
- The discriminating stressor is built: `multi_fact_stressor.py` already has a `--paraphrase` mode that "omits the original keywords" and a marker-bearing `Correction:` fact, so marker-stripping is a config flip, not new infrastructure.

## Approach

Tied to thesis §6 (model the interaction, not the fact) and §4 (the plasticity signal should be learned, not hand-coded):

- **Build the trace corpus.** Run the curriculum through `CortexAgent` and log per-turn `(request, proposed_answer, lm_hidden, actual_outcome ∈ {accepted, corrected})`. Outcome label comes from existing `scoring.py` (`probe_matches`) scoring, not from markers.
- **Self-supervised pretrain (Phase 0 core).** Train `WorldModelCritic.predict_acceptance` teacher-forced on the corpus to predict the binary outcome from the answer-time hidden, with leave-one-episode-out evaluation. This is the §6 "predict acceptance / correction probability before answering" objective, implemented with the MLP that already exists rather than the design-only recurrent SSM.
- **Wire prediction error as the modulator.** At metabolize time the realized error `|predicted_accept − actual_outcome|` (or `correction_likelihood`) becomes the `correction_signal` passed into `cortex.observe()`, replacing `_looks_like_correction`. This closes thesis §6's "correction = prediction error" loop and is the modulator-source single variable.
- **(Secondary, data-permitting)** Add a small correction-*type* head (sense-misgrounding vs scope-overgeneralization) keyed off probe category; report it descriptively, not as a gate, because the 44-episode corpus is thin for multiclass.

## Success criteria

Discriminating, measurable, with kill thresholds. The saturated `code_qa_accuracy=1.0` is explicitly **not** a success metric here; it is reported only to confirm we did not regress it.

| Metric | Success | Kill |
|---|---|---|
| `accept_pred_auc` (world-model, hidden, real driver, leave-one-episode-out) | ≥ 0.70 **and** ≥ string-only critic + 0.05 | ≤ 0.55 (chance) OR not above string-only |
| `marker_free_uptake_gap` = uptake(world-model) − uptake(lexical) on marker-stripped corrections | ≥ 0.40 | < 0.20 |
| `false_plasticity_rate` on non-correction (retention/scope) turns | world-model ≤ lexical | world-model > lexical + 0.10 |
| `modulator_outcome_corr` (point-biserial of modulator vs actual corrected outcome) | > 0 **and** > lexical | ≤ 0 |

Headline = `marker_free_uptake_gap`: it cannot saturate at 1.0 because it is a *difference* between two live arms, and it directly measures the capability the lexical stop-gap lacks.

## Risks & open questions

- **Data scarcity.** 44 episodes + probes is a small corpus for an AUC with confidence; leave-one-episode-out plus reported CIs are mandatory, and the correction-type head may be undertrainable (kept descriptive).
- **Final-layer-only hidden (Goal 2 not done).** The critic reads `peek_embedding` (final-layer mean-pooled), not a mid-network layer-L residual. If acceptance signal lives mid-network, AUC may be capped — this is the explicit cross-link to project 03.
- **Class imbalance.** Initial requests are all corrected; mixed labels come only from probes. Pre-correction AUC must be computed on the probe-balanced subset, not the all-corrected initials, or the number is meaningless.
- **Modulator amplitude vs plasticity clip.** `correction_signal` feeds `alpha_correction=5.0` then clips to [0,1]; a continuous predicted error must be scaled so it spans the useful plasticity band rather than saturating at full overwrite. Open: is a soft (continuous) modulator better than the current binary gate even before semantics?
- **Does prediction error generalize past disambiguation?** Stage 3+ (dialog/cross-domain) were never run through the policy loop; AUC may collapse off the disambiguation manifold.

## Prior evidence

- `WorldModelCritic` already carries acceptance + TD value heads, fed `lm_hidden` every turn but unread as a modulator: `cortex_agent.py:219-226, 463-478`; `world-model-critic/src/world_model_critic/critic.py:171-199, 305-329, 425-440`.
- Lexical stop-gap that this replaces: `cortex_agent.py:53-65` (`_CORRECTION_MARKERS` tuple) tested by `_looks_like_correction` (`cortex_agent.py:68-71`); `cortex_agent.py:347-348` (drives `correction_signal`).
- Phase 0/1 are design-only; gate metrics specified but unrun: `rl_pipeline_design.md` (Phase 0 "predictive foundation", Phase 1 "critic acceptance AUC > 0.8"); RL/policy brief open question "No held-out perplexity / Phase 0 predictive-foundation experiment has been run."
- Value head "trained with TD on every metabolize() … not validated in a real correction/uptake loop" (`src/oczy/experiments/logs/SUMMARY.md`, runs #75–#79).
- Policy/value actor-critic loop runs end-to-end on the real LFM2.5 driver but Stage 2 scope uptake stayed 0/8 (runs #73–#77) — token-overlap terms dominate the small learned signal, motivating a *predictive* gate upstream of the policy rather than more policy capacity.
- Marker ≠ correction edge case: salience run #82 (2026-06-26 salience log) — a marker-free needle turn defeats correction-marker salience.
- Paraphrase recall already implemented and survives keyword omission: run #101 (2026-06-27 paraphrase log), `multi_fact_stressor.py --paraphrase`.
- Saturated headline that must not be the metric: `code_qa_accuracy=1.0` across runs #79,#80,#82,#84,#85,#95,#101.

---
*Cross-links: 01-correction-to-competence-benchmark (label source / curriculum), 05-metabolism-loop-closure (the modulator this feeds), 03-layer-l-hidden-extraction (richer critic input).*
