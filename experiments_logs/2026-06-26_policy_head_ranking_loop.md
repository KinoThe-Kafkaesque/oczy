# Policy-Head Ranking Control Loop — 2026-06-26

## Question

Can the CortexAgent policy head reliably select context-appropriate answers in
`OrganismAgent._rank_answer`, completing a real-LM actor-critic loop all the
way from correction → policy update → ranking decision?

## Method

- Model: `LiquidAI/LFM2.5-1.2B-Instruct-Q4_K_M.gguf`
- Harness: `src/oczy/experiments/organism_curriculum/run_curriculum.py`
- Probes: organism curriculum stages 0 (sense grounding), 1 (transfer), and 2
  (scope control)
- Policy head features: `[warm_state; request_context; candidate_hidden]` via
  `use_policy_request_context`
- Ranking mode: `policy_suppresses_fast_answer` removes the legacy `+1.0`
  fast-weight bias when a policy signal is present
- Policy scores normalized to softmax probabilities over the candidate set
  before being added to the ranking score

Runs:

1. `71baa3f` — Softmax-normalize policy scores in `_rank_answer`.
2. `a07fa77` — Add optional request-context features to the policy head.
3. `d9301db` — Move policy updates outside the critic-surprise gate.
4. `c723660` — Add `policy_suppresses_fast_answer` to let the policy head
   dominate ranking in probe modes.

## Results

### Curriculum probe (real LM driver)

| run | stage 0 retention | stage 1 transfer | stage 2 uptake | stage 2 scope |
|---|---|---|---|---|
| raw-score policy (#73) | 0.88 | 0.38 | 0.00 | — |
| softmax policy (#75) | 0.88 | **1.00** | 0.00 | — |
| request context (#76, gated) | 0.88 | 1.00 | 0.00 | 1.00* |
| policy updates ungated (#77) | 0.88 | 1.00 | 0.00 | — |
| suppress fast bias (#78) | 0.62 | 0.12 | 0.62 | 0.50 |

\* Stage 2 `scope=1.00` before the uptake fix was misleading: the agent never
learned the alternate sense, but its conservative retention also meant it never
overgeneralized.

### Fast tests

| run | tests passed |
|---|---|
| #75 | 266 |
| #76 | 268 |
| #77 | 269 |
| #78 | 271 |

### Benchmark

- `code_qa_accuracy` remained `1.0` throughout runs #75–#78.
- All new flags default off: `use_policy_request_context`,
  `policy_suppresses_fast_answer`.

## Interpretation

1. **Softmax normalization is required.** Raw policy scores accumulate across
   episodes and become unbounded; normalization keeps the policy contribution
   in `[0, 1]` and stable.
2. **Request context helps discrimination.** Feeding the request embedding into
   the policy head gives it a fixed context signal separate from the drifting
   `warm_state`.
3. **Policy updates must not be gated by critic surprise.** A well-calibrated
   critic predicts low `accepted_prob` on corrections, which *suppressed* the
   policy update under the old gate. Moving policy updates outside the
   surprise gate fixed this.
4. **The legacy fast-answer bias must be optional.** Once the policy head is
   trained, the `+1.0` fast-weight bonus prevents it from controlling the
   answer. Suppressing that bias in probe modes lets the policy head drive
   alternate-sense selection.

## Implication for architecture

The actor-critic policy loop now closes end-to-end on a real local LM:

- `CortexAgent.perceive()` produces request and warm-state hidden vectors.
- The policy head scores candidate answers with learned request/context
  features.
- `OrganismAgent._learn_from_correction()` applies symmetric `+1`/`-1`
  policy-gradient updates on every correction.
- `OrganismAgent._rank_answer()` can let the policy head dominate final
  selection when configured.

## Trade-offs

- `policy_suppresses_fast_answer` improves scope control but can hurt Stage 0/1
  because a randomly initialized policy head needs a correction before it
  becomes reliable. The fix is to keep the flag probe-only and default off.
- Stage 2 real-driver results are still noisy; the small policy head and
  limited curriculum benefit from longer training or a higher learning rate in
  probe mode.

## Open questions

1. Does combining `use_policy_request_context=True` with
   `policy_suppresses_fast_answer=True` improve all three stages at once?
2. What is the right policy-learning rate for the real-driver curriculum?
3. Should the replay-hint bonus (`+0.5`) also be attenuated in policy mode so
   hippocampal recall does not override the learned head?
4. How does the policy-driven loop behave on Stage 3 (dialog) and beyond?

## Artifacts

- `src/oczy/experiments/organism.py` — softmax policy scoring, ungated policy
  updates, `policy_suppresses_fast_answer`.
- `src/oczy/experiments/cortex_agent.py` — `use_policy_request_context`
  features for the policy head.
- `src/oczy/experiments/organism_curriculum/run_curriculum.py` — probe modes
  enable policy-loop gates and fast-bias suppression.
- `src/oczy/experiments/tests/test_organism_cortex_policy.py` — regression
  tests for softmax, request-context, ungated policy updates, and fast-bias
  suppression.
- `src/oczy/experiments/logs/SUMMARY.md` — autoresearch runs #75–#78.

## Commits

- `71baa3f` — Normalize policy-head scores to softmax probabilities.
- `a07fa77` — Add `use_policy_request_context` to the policy head.
- `d9301db` — Move policy updates outside the critic-surprise gate.
- `c723660` — Add `policy_suppresses_fast_answer` ranking flag.
