# LM progression

Found and fixed a 1-step-delayed hidden-state bug in `_rnn_forward`
(`plastic-cortex/src/plastic_cortex/_numba_kernels.py:75-115`).
The forward stored `hiddens[t]` as the *pre-step* state and computed
`logits[t]` (the prediction of `tokens[t+1]`) from a hidden state that
hadn't yet seen `tokens[t]`.  The backward pass was written under the
opposite convention (post-step), so the gradient for `dE`/`dW_xh` was
assigned to the wrong token id by 1 position.  This caps the LM at
unigram-with-1-step-delay = exactly the ~3.2-nat plateau observed across
every existing checkpoint (`lm_bpe_500_full` plateaued at 5.28, etc.).

# Result

The 40K-parameter LM (`LMPlasticCortex`, hidden_dim=128, char tokenizer,
103-token vocab) climbed all 6 stages of the graduated curriculum
(`plastic-cortex/data/progression/`) and reached the project's actual
goal: quoted-word disambiguation.

| Stage | Best loss | Top-1 next-char | Promoted |
|---|---:|---:|---|
| 0 char n-gram calibration | 0.595 | 0.777 | yes |
| 1 copula "X is Y" | 0.950 | 0.690 | yes |
| 2 class properties | 0.854 | 0.641 | yes |
| 3 multi-clause syllogism | 0.701 | 0.652 | yes |
| 4 dialog | 0.972 | 0.541 | yes |
| 5 quoted-word disambig | 1.010 | 0.345 | yes |

Random baseline is 1/103 = 0.0097.  Stage 0 hits 80x random; stage 5
is 36x random.  Per-stage loss is per-token cross-entropy (mean
across held-out positions), so `e^0.595 = 1.8` effective characters at
stage 0 (down from `e^4.63 = 103` uniform) and `e^1.01 = 2.7` at
stage 5 (the model genuinely cannot predict exact disambiguating quotes
but it has internalised the "X means Y" template far better than
chance).

# What broke

1. **Architecture bug.**  `_rnn_forward` returned `hiddens[t] = pre-step
   state` and computed logits the same way; the backward pass expected
   `/post-step convention`.  One-line fix: move `_rnn_step` to the top of
   the loop and store `hiddens[t] = h` AFTER the update.

2. **Flat curriculum.**  The original `CURRICULUM_TEMPLATES` mixed
   "hello" with "Explain the Oczy organism in one sentence" — a
   10,000x difficulty spread.  Replaced with a 6-stage cumulative
   ladder where vocab grows monotonically: 18 -> 29 -> 41 -> 48 -> 63
   -> 86 words.  Stages 0-2 are bigram/template drills; stage 3 is
   clause-level syllogism chaining; stage 4 is simple turn-taking;
   stage 5 is the actual project goal (quoted-word disambiguation).

3. **Training instability.**  `lm_assistant_1k_stable` had logged a
   blow-up (3.13 -> 3.58 -> 125 -> 341).  Driver now has: strict
   grad-clip, NaN/divergence detection with checkpoint restore + LR
   halving, plateau detection that decays LR rather than stopping.
   Effectively Schedule: 0.02 -> 0.01 -> 0.005 -> 0.0025 -> 0.00125
   (4 LR halvings with max_lr_halvings = 3 for hard stages; stage 5
   needed the 4th halving at lr=0.0006).

# What didn't work / open gaps

- **Greedy-generation exposure bias.**  Samples for stages 1-5 still
  look mostly like gibberish (`'a rose is a' -> 'ark.'`,  `'"branch" means' -> ' an an today.'`)
  even though the held-out top-1 is 0.69 across these stages.  The
  underlying distribution is competent but autoregressive argmax
  drifts off the learned manifold.  Top-p / nucleus sampling or
  scheduled-sampling during training (mixing teacher-forced and
  model-fed positions) would close the gap.

- **Stage 5 is the practical ceiling at 40K params.**  The 12 quoted
  disambiguations (`branch` -> `git`, `batch` -> `ml`, ...) each
  require exact recollection of a specific stimulus→response pair
  without much shared structure.  Loss plateaued at 1.01 with held-out
  top-1 = 0.345 — well above chance but a clear capacity wall.

- **The benchmarks still measure the wrong thing.**  The 12-episode
  word-sense benchmark in `correction_benchmark` still has
  token-leakage in its transfer probes and trivialises its
  forgetting probes (see `NOTES.md`).  Climbing the LM ladder
  doesn't fix that the benchmark overstates PlasticCortex's score
  due to token overlap.

# Reproduce

```
$ rm -rf plastic-cortex/checkpoints/lm_progression/*    # cold start
$ uv run python experiments/lm_progression/run_progression.py \
      --epochs-per-stage 80 --lr 0.02 --grad-clip 5.0 --max-stages 6
# Stage 5 (hardest) benefits from starting at lower LR with more patience:
$ uv run python experiments/lm_progression/run_progression.py \
      --epochs-per-stage 200 --lr 0.005 --grad-clip 5.0 \
      --from-stage 5 --max-stages 1
```

Reports under `experiments/lm_progression/reports/stage_0[0-5]_*.md`
and `experiments/lm_progression/reports/stage_0[0-5]_*.json`.

# Total wallclock

Stages 0-4 took ~2 minutes combined (each plateaus within 80 epochs
of training on a 100-300-line corpus).  Stage 5 alone took ~40
seconds at lr=0.005 starting from stage 4's checkpoint.