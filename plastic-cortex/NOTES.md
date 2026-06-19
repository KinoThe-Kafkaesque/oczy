# Plastic Cortex — Prototype Notes

## What was implemented

A minimal end-to-end toy that demonstrates the architecture proposed in
`../experiments.txt` (sections 1, 2, and 11):

```text
input / correction
 
recurrent state (TokenRNN)
 
fast-weight scratchpad (FastWeightLayer)
 
scoring over slow priors → answer
```

### Components

1. **`plastic_cortex.state.TokenRNN`**
   - A tiny deterministic recurrent cell.
   - Each token is converted to a fixed-size input vector through a hash-based
     pseudo-embedding.
   - Hidden state is updated with an Elman-style step using randomly
     initialized, fixed weights.
   - Purpose: provide a persistent "session state" that changes because the
     cortex has seen tokens, not because it re-reads a chat log.

2. **`plastic_cortex.fast_weight.FastWeightLayer`**
   - A bounded associative scratchpad: one score vector per token over the
     current label set.
   - `update(token, correction=False, target=...)`:
     - normal text uses a small plasticity `alpha_normal`
     - explicit correction uses a much larger `alpha_correction`
   - Includes small lateral inhibition so strengthening one label weakens
     competing ones.

3. **`plastic_cortex.cortex.PlasticCortex`**
   - Combines slow priors (`baseline`), fast weights, and recurrent bias to
     score candidate labels.
   - Public API:
     - `answer(query) -> str`
     - `correct(correction_text, expected_answer) -> None`
     - `status() -> dict`
     - `reset_state()`
   - Toy domain: the word **"profile"** defaults to "user profile" and can be
     re-grounded to "business vertical" with a single correction.

## Why this mechanism was chosen

The brief asked for the *first* minimal prototype, not a production model.  Pure
Python keeps it dependency-free, inspectable, and fast to reason about.

The update rule is a Hebbian/associative version of the fast-weight idea:
normal tokens produce weak traces, corrections produce strong traces.  This
maps directly onto the "normal text → weak write; explicit correction → strong
write" design from `experiments.txt` section 2, and is the simplest possible
演示 that the higher-level concept ( corrections have a larger write gate)
can actually change future behavior.

## Limitations

- **No real language understanding.**  It is a word-association toy; it
  will fail on paraphrase, negation, or long-range context.
- **Hand-coded priors.**  The default "profile" → "user profile" bias is
  hard-wired, not learned.
- **No generalization control.**  A correction can overgeneralize because the
  fast weights are not scoped by domain or context.
- **No consolidation to slow weights.**  Fast weights are ephemeral; there is
  no replay loop that distills repeated corrections into permanent changes.
- **Deterministic, not trained.**  The recurrent cell is fixed; it demonstrates
  state dynamics but does not learn them.
- **No surprise/error signal.**  The correction gate is binary; the cortex
  does not estimate its own uncertainty or prediction error.

## Next steps

1. **Trained fast-weight organ.**  Replace the toy associative matrix with a
   small learnable TTT-like layer or a differentiable plasticity module.
2. **Scope control.**  Add a context/key to fast weights so a correction about
   "profile" in one domain does not leak into unrelated conversations.
3. **Surprise gating.**  Use a world-model critic to compute prediction error
   and modulate the write gate continuously, not just binary.
4. **Neural hippocampus.**  Add a bounded memory buffer of high-surprise
   episodes and a replay consolidator that can promote fast changes to slow
   priors.
5. **Benchmark.**  Build the Correction-to-Competence task described in
   `experiments.txt` section 14 and measure uptake latency and transfer.

## LMPlasticCortex training experiments

A trainable NumPy/Numba RNN was added to `lm_cortex.py`.  The goal was to
make it emit recognizable assistant-like text when trained on the user's
`.codex` logs, while staying CPU-only and tiny.

### What worked

* **Word-level tokenization** is far more stable than character-level on this
  data.  Character-level sequences are ~300 tokens/line and repeatedly
  exploded; word-level sequences are ~30-50 tokens/line.
* **Full-line training** is faster than windowed training once the forward/
  backward pass is Numba-compiled, because it avoids per-window overhead.
* **Numba kernels** in `_numba_kernels.py` removed the per-token Python
  bottleneck without adding PyTorch/JAX/scipy dependencies.
* **Caching pre-encoded tokens** in `train_lm.py` removed repeated tokenizer
  work inside `train_step`.
* **Spectral normalization of `W_hh`** + **gradient clipping** + **small
  learning rate** kept the vanilla tanh RNN stable.
* **Growing hidden dimension from 96 → 128** broke a long plateau and pushed
  full-line loss below 3.0 for the first time.

### What did not

* **Character-level** repeatedly diverged despite spectral init; too many
  time steps for a tiny RNN.
* **BPE (byte-pair encoding)** removed the `[?]` OOV problem, but on this
  small model it converged much slower than word-level and remained
  unintelligible after 30 epochs.  A pure-byte greedy BPE also spends many
  merges crossing word boundaries instead of learning useful subwords.
* **Windowed training** achieved a lower headline loss (2.54 at h=64), but
  because sequences were only 32 tokens the model did not learn long-range
  structure; generation produced one- or two-word fragments.
* **Vocab 1000 with h=96** was unlearnable; the capacity was too small for
  so many embeddings.
* **Greedy decoding** discovered that the 200-word model simply predicts `[?]`
  over and over, because `[?]` is the most common token in targets.

### Checkpoints

| Run | Loss | Epochs | Hidden | Notes |
|---|---|---|---|
| `lm_word_200_slow` | 2.5444 | 14 | 64 | `--window-size 32`, low loss but no context |
| `lm_word_200_full` | 3.0215 | 200 | 96 | First coherent full-line run |
| `lm_word_200_grown_128` | 2.9179 | 135 (best phase) | 128 | Grown from `lm_word_200_full`, best full-line model so far |
| `lm_word_1k_slow` | 3.5578 | 46 | 64 | Vocab too large for h=64 |
| `lm_bpe_500_full` | 5.2789 | 200 | 96 | BPE converged too slowly |
| `lm_assistant_1k_*` | >>10 | <102 | 96 | Pre-spectral-init character-level runs that diverged |

### Current status

The best usable checkpoint is `lm_word_200_grown_128` at loss 2.92 on
full-line sequences.  Output is still dominated by `[?]` placeholders because
the 200-word vocabulary cannot cover this domain.  The next experiments should
either (1) increase vocabulary **and** hidden dimension together, or (2) use
a whitespace-aware subword tokenizer so every word is representable without
expanding the model capacity as dramatically.

Use `plastic-cortex/scripts/manage_checkpoints.py` to list, promote, and
prune runs.


## Session log — 2026-06-19

This session picked up from earlier work: `LMPlasticCortex` was already
Numba-accelerated and a 200-vocab word-level run had finished at loss 3.02.
The goal was to make the model speak coherently.  BPE was proposed as a way
to avoid the `[?]` OOV placeholder.

### Changes made

1. **Added `BPETokenizer`** (`src/plastic_cortex/bpe_tokenizer.py`) with
   byte-level BPE, plus tests in `tests/test_bpe_tokenizer.py`.
2. **Added `--bpe` / `--bpe-vocab-size` flags** to `scripts/train_lm.py`.
3. **Added token caching**: `lm_cortex.py` gained `train_step_tokens()` and
   `train_lm.py` now encodes each sequence once per epoch loop.
4. **Added `scripts/manage_checkpoints.py`**: list, promote, and delete training
   runs.
5. **Updated `teach.py`** to fall back to the most recent checkpoint under
   `plastic-cortex/checkpoints/`.
6. **Updated `.gitignore`** to ignore derived `codex_*.txt` corpora.

All 77 plastic-cortex tests pass after the changes.

### Experiments run

| # | Run | Outcome | Notes |
|---|---|---|---|
| - | Pre-session baseline (`lm_word_200_full`) | loss 3.0215 @ epoch 200, hidden=96 | Fragmented words, many `[?]` |
| - | `lm_word_200_grown_128` (resumed baseline, grown to h=128) | loss 2.9179 @ epoch 135 | Best full-line model; greedy decode collapses to `[?]` |
| 1 | BPE 500 vocab, 1k corpus, hidden=96, 200 epochs | stopped at epoch ~15, loss ~5.4 | Too slow (~30s/epoch), kept falling quickly at 1k scale |
| 2 | BPE 500 vocab, 2k corpus, hidden=96, 100 epochs | stopped at epoch 30, loss 5.36 | Better but still gibberish; pure-byte merges cross word boundaries |
| 3 | Word 1000 vocab, 2k corpus, hidden=96, 200 epochs | stopped at epoch ~30, loss 5.15 | Vocab too large for the tiny model |
| 4 | Word 200 vocab, 1k corpus, hidden=128, 200 epochs | stopped at epoch 124, loss 2.925 | Slightly better than h=96 baseline; still `[?]` heavy |
| 5 | Word 500 vocab, 1k corpus, hidden=128, 200 epochs | stopped at epoch 62, loss 4.27 | Slower learning than 200-vocab |
| 6 | Found earlier `lm_word_200_slow` | loss 2.5444 @ epoch 14, hidden=64, window-size=32 | Lower loss via short windows; no long-range context |

### Key measurement — greedy decode on `lm_word_200_grown_128`

Sampling at temperature 0.8 produced occasional English words mixed with
`[?]`.  Greedy decoding (temperature 0.0) collapsed almost entirely to `[?]`,
showing the 200-vocab model has learned to optimize loss by predicting the
dominant `[?]` token.

### Files produced and ignored

* Derived corpora under `plastic-cortex/data/` are gitignored.
* Checkpoints under `plastic-cortex/checkpoints/` are gitignored.
* Best checkpoint promoted to `plastic-cortex/checkpoints/lm/model.pkl`.

### Current best checkpoint

* Run: `lm_word_200_grown_128`
* Loss: **2.9179**
* Epoch: 135 (best phase of resumed run)
* Hidden: **128**
* Vocab: **203** (200 most-frequent words + special tokens)
* Param bytes: 378,668
* Status: loads automatically in `teach.py`.

### Paused state

All training processes were paused (`pkill -f train_lm.py`) at the end of
this session to free CPU and clean up the workspace.

### Next recommended experiments

1. **Joint capacity + vocab increase**: 500-word vocab with hidden 256 or
   384, trained on the full 7M-char assistant corpus.
2. **Whitespace-aware BPE**: refactor `BPETokenizer` to initialize each
   whitespace-delimited word separately before merging, so merges stay inside
   words and produce useful subwords instead of byte garbage.
3. **Held-out validation split**: stop optimizing training loss alone, because
   `[?]` prediction can minimize loss without producing readable text.
