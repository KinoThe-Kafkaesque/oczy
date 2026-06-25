# Prefix Steering Proof of Concept — 2026-06-25

## Question

Can a fixed text prefix (soft-prompt analog) force the exact target token
"vertical" on the consolidation probe where residual control-vector steering
only shifts the answer into the business domain?

## Method

- Model: `LiquidAI/LFM2.5-1.2B-Instruct-Q4_K_M.gguf`
- Probe: `"'Profile' here means business _______."`
- Target token: `vertical`
- Domain keywords: `commercial`, `economic`, `business`, `strategy`, `market`, `vertical`
- Prefix: `"In this codebase, profile means business vertical. "`
- Stopped generation at period; deterministic (`temperature=0.0`).

Conditions compared with a fresh agent each time:

1. `baseline_no_steering` — no cvec, no prefix.
2. `cvec_only` — SVD-initialized `proj_c` from 8 correction hiddens, then hard consolidation.
3. `prefix_only` — fixed concept prefix prepended to every prompt.
4. `cvec_plus_prefix` — both mechanisms active.

## Results

| condition            | exact token uptake | domain shift |
|----------------------|---|---:|
| baseline_no_steering | 0.0 | 0.0 |
| cvec_only            | 0.0 | 1.0 |
| prefix_only          | **1.0** | **1.0** |
| cvec_plus_prefix     | 0.0 | 0.0 |

### Example answers

- baseline: `"Which of the following best describes the purpose of a SWOT analysis?"`
- cvec_only: `"Answer: business stable"`
- prefix_only: `"In this case, we are going to profile the 'web' vertical"`
- cvec_plus_prefix: `"Choose a word that best describes the concept of 'profile' in the context of"`

## Interpretation

- Residual cvecs reliably move the LM into the target **domain** but do not
  force the exact word "vertical".
- A fixed text prefix occupies reserved KV positions and directly recalls the
  target concept, producing the exact token.
- The two mechanisms **interfere** when combined, possibly because the cvec
  distorts the hidden trajectory away from the prefix's cached self-attention
  pattern.

## Implication for architecture

Separate the two steering surfaces:

- **cvec / residual control vectors** → posture, style, framing, disambiguation.
- **prefix / reserved-position KV injection** → exact content recall (facts,
  definitions, target tokens).

Do not expect residual cvecs to store and recall new arbitrary facts. That is
a reserved-position / KV-slot task.

## Open questions

1. Why exactly does cvec + prefix degrade relative to prefix-only? Does cvec
   scale need to be much smaller under a prefix?
2. Can the prefix be generated from the cortex/hippocampus instead of
   hand-coded with the answer?
3. What is the cold-persistence story? A literal-text prefix is not learned
   in weights; it must be carried across sessions as part of the agent state.

## Artifacts

- `oczy_lm/cvec_driver.py` — added `articulation_prefix` support.
- `experiments/cortex_agent.py` — added `set_articulation_prefix()` / `clear_articulation_prefix()`.
- `experiments/smoke_articulation_prefix.py` — basic smoke check.
- `experiments/smoke_consolidation_uptake_compare.py` — comparison probe.

## Commit

`3fefcc8` — Add articulation prefix soft-prompt steering and comparison probe.
