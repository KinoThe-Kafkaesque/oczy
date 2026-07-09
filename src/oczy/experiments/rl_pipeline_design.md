# RL Pipeline for the Oczy Organism

**Goal:** evolve the current Plastic World Model Agent from a correction-gated word-association toy into an agent that exhibits *meaningful thought*: sustained internal modeling, planning, self-correction, and transfer across tasks.

**Status:** the architecture in `experiments.txt` is correct in spirit, but every organ is currently a hand-coded prototype. This document describes a staged reinforcement-learning pipeline that replaces those prototypes with learnable components while keeping the organism's structural shape (cortex, hippocampus, critic, hypernetwork, immune system, autoencoder).

---

## What "meaningful thought" means here

We operationalize it into observable capabilities:

| Capability | Observable behavior |
|---|---|
| Internal world model | Agent can predict the likely user response / correction before it happens. |
| Planning | Agent considers multiple candidate responses and selects the one with best predicted outcome. |
| Self-correction | Agent detects its own mistake from user feedback and changes behavior after one correction. |
| Transfer | A correction on one ambiguous token improves performance on semantically related probes. |
| Scope control | The correction does not degrade unrelated behavior. |
| Curiosity | Agent asks clarifying questions when uncertainty is high, without being told to. |
| Skill abstraction | Repeated solution patterns crystallize into reusable options. |
| Reflection | After failure, agent generates a compact lesson and updates its identity. |

We do **not** claim this pipeline produces consciousness or general intelligence. It produces a *functionally thoughtful* agent on a bounded task domain.

---

## The RL framing

Treat interaction as a partially-observable Markov decision process (POMDP):

*   **State** `s_t`: hidden recurrent state of the organism + memory contents + identity latent.
*   **Observation** `o_t`: user message / task description + tool outputs.
*   **Action** `a_t`: a structured response chosen from a finite action space.
    In early phases the action space is just text labels; in later phases it expands to include internal "thought" tokens, clarification questions, and tool calls.
*   **Reward** `r_t`: shaped from user acceptance, task completion, correction feedback, prediction error, and novelty.
*   **Policy** `π(a_t | h_t)`: the organism's response distribution conditioned on history.
*   **Value** `V(h_t)`: expected cumulative return from history.

The central training objective is:

$$
\mathcal{L} = \mathbb{E}_{\tau} \left[ \sum_{t} \gamma^t r_t \right]
$$

plus auxiliary losses that keep components honest (prediction, compression, anti-overgeneralization).

---

## Staged pipeline

Each phase trains one new capability. Phases are sequential because each provides the inductive bias for the next.

---

### Phase 0 — Predictive foundation (self-supervised world model)

**What:** Before acting, the agent must model the conversation dynamics.

**Objective:** on a corpus of interaction traces

$$
\mathcal{L}_{\text{pred}} = -\log P(x_{t+1} \mid h_t, x_t)
$$

where `x_t` is the current token/message and `h_t` is the recurrent state.

**Data:** transcripts from `correction-benchmark` extended with synthetic user turns.

**Algorithm:** teacher-forced next-step prediction.

**Organs affected:**
*   `plastic_cortex` becomes a recurrent language/state-space model instead of a word-association table.
*   `world_model_critic` seeds its features from the predictive cortex.

**Success criterion:** perplexity on held-out conversation traces decreases; the cortex produces a hidden state that encodes which ambiguous sense was last active.

---

### Phase 1 — Outcome predictor (the learned critic)

**What:** turn the hand-coded `WorldModelCritic` into a learned value/return model.

**Objective:** predict whether the agent's answer will be accepted and estimate cumulative future return.

$$
\mathcal{L}_{\text{critic}} = \left( R_t - V(h_t) \right)^2 + \text{BCE}(\hat{a}_t, a_t)
$$

where `R_t` is the observed return, `V(h_t)` the value estimate, `a_t` the binary acceptance signal.

**Data:** curriculum rollouts where the reward is revealed after each response.

**Algorithm:** temporal-difference learning (TD(λ)) on accepted/corrected outcomes.

**Organs affected:**
*   `world_model_critic` gains both an acceptance head and a value head.
*   The critic's `record_outcome()` stores `(h_t, a_t, r_t)` tuples for replay.

**Success criterion:** critic's acceptance AUC > 0.8 and value predictions correlate with actual return.

---

### Phase 2 — Response policy with REINFORCE

**What:** train a policy that maps history to responses, optimizing for acceptance + task reward.

**Objective:**

$$
\mathcal{L}_{\text{policy}} = -\mathbb{E}_{\pi} \left[ \sum_t \left( R_t - V(h_t) \right) \log \pi(a_t \mid h_t) \right]
$$

**Data:** rollouts on the Correction-to-Competence curriculum.

**Algorithm:** REINFORCE with baseline (actor-critic). The fast-weight organ stores policy updates session-local.

**Action space:** for Phase 2, the agent emits one of the corrected answer labels and optionally a confidence score. Later phases expand to full text generation.

**Reward shaping:**
*   `+1` answer accepted immediately.
*   `+0.5` answer correct after one correction.
*   `-0.2` answer still wrong after correction.
*   `-1` overgeneralization (wrong on a scope probe).

**Organs affected:**
*   `plastic_cortex` now has a policy head that uses fast weights.
*   `neural_hippocampus` stores high-surprise trajectories, not just episodes.
*   `identity_hypernetwork` receives policy-gradient-derived updates.

**Success criterion:** the agent reaches correction-uptake latency < 0.3 and transfer score > 0.5 on the curriculum.

---

### Phase 3 — Correction as a learning signal

**What:** corrections become first-class RL events that trigger rapid plasticity, not just negative reward.

**Objective:** learn a *plasticity modulator* `η_t` such that the update `φ_{t+1} = φ_t - η_t_φ L(a_t, c_t)` is itself learned.

$$
\mathcal{L}_{\text{plasticity}} = \mathbb{E}\left[ \mathcal{L}(a_{t+1}, y^*) \right] + \lambda \|\eta\|_2^2
$$

After a correction at turn `t`, evaluate the agent again at turn `t+1`; the plasticity network is trained to make `t+1` correct.

**Data:** pairs of (wrong answer, correction, follow-up probe) from the curriculum.

**Algorithm:** differentiable plasticity / meta-gradient on the fast-weight update rule.

**Organs affected:**
*   `plastic_cortex` fast weights gain a learned plasticity coefficient per parameter.
*   `skill_immune_cortex` learns to gate plasticity (don't learn when input is noisy or contradictory).

**Success criterion:** one-shot correction uptake reaches > 0.8; contradictory corrections do not cause catastrophic failure.

---

### Phase 4 — Intrinsic motivation (curiosity)

**What:** the agent seeks information, not just reward.

**Objective:** augment reward with a prediction-error bonus:

$$
r_t = r_t^{\text{env}} + \beta \cdot \|\hat{o}_{t+1} - o_{t+1}\|^2
$$

where `β` is a novelty coefficient that decays as the world model improves.

**Action expansion:** add a `clarify()` action. Asking a targeted question yields intrinsic reward if the answer reduces critic uncertainty.

**Algorithm:** curiosity-driven exploration with an episodic curiosity module (e.g. random network distillation or hash-count).

**Organs affected:**
*   `world_model_critic` maintains a separate "prediction-error" head.
*   `neural_hippocampus` stores novelty counts.
*   `plastic_cortex` response distribution gains a "question" branch.

**Success criterion:** on ambiguous probes the agent asks clarifying questions when confidence is low, and its acceptance rate improves with fewer corrections.

---

### Phase 5 — Hierarchical skills / options

**What:** crystallize repeated interaction patterns into reusable skills.

**Option framework:**
*   A *meta-policy* selects an option (e.g. `disambiguate_token`, `check_constraint`, `summarize`).
*   Each option is a sub-policy that runs until termination.
*   Termination is learned.

**Objective:** option-critic loss

$$
\mathcal{L}_{\text{option}} = \mathcal{L}_{\text{intra}} + \mathcal{L}_{\text{termination}} + \mathcal{L}_{\text{policy-over-options}}
$$

**Data:** replay buffer grouped by recurring task structures (SQL safety review, disambiguation, refactoring).

**Organs affected:**
*   New `skill_cortex` module learns option policies.
*   `experience_autoencoder` encodes option executions into compact Δz updates.
*   `skill_immune_cortex` detectors become option preconditions.

**Success criterion:** transfer across structurally similar tasks improves; policy avoids recomputing the same reasoning steps.

---

### Phase 6 — Meta-RL for fast adaptation

**What:** train the slow weights so that the agent adapts quickly inside a new session.

**Objective:** MAML-style outer loss

$$
\min_\theta \sum_i \mathcal{L}_i\left( \theta - \alpha \nabla_\theta \mathcal{L}_i^{\text{inner}} \right)
$$

**Inner loop:** fast-weight updates on the first few turns of task `i`.
**Outer loop:** gradient through the adapted parameters on remaining turns.

**Organs affected:**
*   `plastic_cortex` slow weights become the meta-initialization.
*   `identity_hypernetwork` generates task-specific priors used in the inner loop.
*   `neural_hippocampus` provides task-specific replay.

**Success criterion:** on a held-out task, the agent fixes its first mistake in one correction and transfers to related probes without training on them.

---

### Phase 7 — Internal simulation / "thought"

**What:** before answering, the agent simulates candidate responses internally.

**Mechanism:** add a *thinking* action that produces a latent trajectory without committing to an external response. The world model rolls out imagined user reactions; the best imagined trajectory is selected.

**Objective:** learn a *think policy* that minimizes predicted regret.

$$
\mathcal{L}_{\text{think}} = \max_a Q(h_t, a) - Q(h_t, a^*) + \lambda \cdot \text{length}(\text{thought}_t)
$$

The thought length penalty keeps thinking bounded.

**Organs affected:**
*   `plastic_cortex` gains a separate "internal roll-out" mode.
*   `world_model_critic` is trained on imagined trajectories via Dyna-style planning.
*   `experience_autoencoder` compresses successful thought patterns.

**Success criterion:** the agent refuses or asks clarifying questions on genuinely ambiguous inputs; performance improves on adversarially perturbed probes.

---

### Phase 8 — Self-curriculum and autonomous goals

**What:** the agent proposes tasks to itself and practices them.

**Loop:**
1. Identify weakest skill by measuring recent prediction error.
2. Generate synthetic training episodes that target that skill.
3. Run practice rollouts.
4. Consolidate only if practice improves validation score.

**Algorithm:** curriculum learning + self-play with the frozen prior as a critic.

**Organs affected:**
*   `skill_immune_cortex` defines skill-level regression tests.
*   `identity_hypernetwork` stores the current curriculum focus.
*   `neural_hippocampus` rejects synthetic data that increases forgetting.

**Success criterion:** continual improvement on a growing benchmark without manual retraining.

---

## Component upgrades required

| Organ | Current | Upgrade for RL |
|---|---|---|
| `plastic_cortex` | Hand-coded word-association RNN | Recurrent / SSM policy network with separate policy, value, and prediction heads. |
| `world_model_critic` | Logistic-regression acceptance classifier | Learned value + outcome predictor trained with TD(λ). |
| `neural_hippocampus` | Episode store with similarity retrieval | Trajectory replay buffer with prioritized experience replay; supports policy gradients. |
| `experience_autoencoder` | Compresses episodes to Δz | Compresses *trajectory segments* to gradient-shaped Δz; trained with behavior reconstruction + compression loss. |
| `identity_hypernetwork` | Generates tiny concept-score deltas | Generates LoRA-style adapter weights from latent `z`; meta-trained for fast adaptation. |
| `skill_immune_cortex` | Hard-coded mistake detectors | Learned constraint / detector network that blocks policy updates violating old skills. |

---

## Training loop (master algorithm)

```text
for phase in [0, 1, ..., 8]:
    for epoch:
        for episode in curriculum.sample():
            for turn:
                h_t = organism.observe(obs_t)
                a_t ~ policy(h_t)
                obs_{t+1}, r_t, done = env.step(a_t)
                hippocampus.store((h_t, a_t, r_t, obs_{t+1}, done))
                critic.update(h_t, a_t, r_t, obs_{t+1})

            policy.update(trajectory)
            train_auxiliaries(world_model_prediction, compression, anti-forgetting)

        if phase >= 6:  # meta-RL
            meta_update()

        if phase >= 7:  # imagination
            dyna_planning_update()

    consolidate()
    immune.check_regression()
```

Auxiliary losses:
*   `L_prediction` — world-model accuracy.
*   `L_compression` — small latent size.
*   `L_replay_retention` — old replay buffer still scored correctly.
*   `L_scope` — scope probes do not drift.

---

## Success metrics per phase

We do **not** optimize a single loss. We track a dashboard:

| Phase | Gate metric | Target |
|---|---|---|
| 0 | Held-out prediction perplexity | < baseline by 20% |
| 1 | Critic acceptance AUC | > 0.8 |
| 2 | Curriculum transfer score | > 0.5 |
| 3 | One-shot uptake | > 0.8 |
| 4 | Clarification rate when uncertain | > 0.6 |
| 5 | Skill reuse on similar tasks | > 0.6 |
| 6 | Few-shot adaptation on new task | uptake after 1 correction > 0.8 |
| 7 | Internal roll-out regret reduction | > 20% vs. no-simulation baseline |
| 8 | Self-curriculum validation improvement | positive trend over 1000 self-generated episodes |

The north-star metric remains:

```text
behavior_delta_per_byte_of_persistent_memory
```

---

## Concrete first implementation steps

1. Replace `PlasticCortex` with a small recurrent network that can be policy-trained.
2. Add a `Trajectory` data structure and `ReplayBuffer` to `NeuralHippocampus`.
3. Split `WorldModelCritic` into acceptance head and value head.
4. Implement `RLTrainer` that runs the Phase 0–2 loop on the Correction-to-Competence curriculum.
5. Add a uniform reward-shaping layer that maps user outcome to scalar reward.
6. Before adding creativity components, harden evaluation: every policy update must pass the immune-system regression test.

---

## Risk: current organ prototypes are not trainable

The existing workspace packages are intentionally hand-coded toys. Applying this pipeline verbatim will fail because the gradients do not flow through `plastic_cortex.correct()` or `identity_hypernetwork.update_identity()` in a differentiable way. The pipeline presumes each organ is reimplemented as a neural module with trainable parameters. This document is the roadmap; the next step is to decide which organ to make differentiable first (recommended: `plastic_cortex`, because it is the policy substrate; then `world_model_critic`, because it provides the reward signal).

---

## Summary

The path to meaningful thought is not a single trick. It is a staircase where each step adds one missing capacity: prediction, value estimation, policy gradient, learned plasticity, curiosity, skills, meta-adaptation, internal simulation, and self-curriculum. The Oczy organism already has the right organ names; this pipeline describes how to make those organs trainable and arrange them into a coherent RL agent.
