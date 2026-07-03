# Note — Where the steering intuition was wrong (and what survives)

**Date:** 2026-07-03
**Status:** conceptual post-mortem, written after the Sprint 1–3 refutation
arc. Not an experiment log; the evidence lives in the logs cited inline.

## The intuition under examination

*Experience should steer the model's activations: embed each interaction,
accumulate the embeddings into a fast-weight state, emit that state as
residual-stream vectors (cvecs), and the model behaves differently in the
direction of its experience.*

The thesis behind it — interactions should become changed processing, not
retrieved content — is untouched by anything below. Activation steering is a
real, published phenomenon. The failure is in three specific assumptions that
were bundled into the implementation, each now isolated experimentally.

## Assumption 1 — accumulated experience has a *direction*. It mostly has a *magnitude*.

Averaging correction embeddings amplifies what corrections share — the
format ("No, 'X' means Y"), the syntax, the register — not what distinguishes
them. The content is the minority signal. Summing K episodes therefore grows
the common mode: the vector gets *louder* with every interaction while
pointing at "correction-ish text" rather than at any particular fact.

**Evidence:** S2.4 (`2026-07-01_s2_4_breakthrough_ablation.md`) — control
words rose MORE than target words (Δ_control 2.92 vs Δ_target 1.60); clamping
the norm to a fixed budget erased the entire claimed gain.

**Contrast with published steering:** contrastive construction
(with-property activations minus without-property activations) cancels the
common mode by design. The Hebbian accumulation never performed that
subtraction.

## Assumption 2 — a constant vector can carry *conditional* content. It structurally cannot.

A fact like "'log' means the captain's journal" is an if-then: *when* 'log'
appears in the right context, produce different tokens. A residual-stream
addition is prompt-independent — a rank-1 global bias that says "always be
more X-ish," applied identically to every token of every input. It can shift
low-dimensional, unconditional attributes (style, sentiment, domain-prior —
genuine *posture*), but it has no query, so it cannot condition.

**Evidence:** the 06-27 finding — cvec-forced target tokens stuck at rank
~47,000 while a text prefix and KV-splice (S1.3,
`2026-07-01_s1_3_hf_kv_slot_injection.md`) hit rank 1: exact recall needs
content routing through attention, not bias. The 06-25/06-27 logs had already
honestly labeled cvecs "posture surface only"; the architecture then kept
asking posture machinery to carry content.

## Assumption 3 — the embedding of a correction ≈ the activation direction that would *use* it.

`peek_embedding` reads how the model represents a sentence *about* a fact.
Injecting that back assumes mention-space and use-space coincide — that the
direction excited by reading "'log' means journal" is the direction that
would make the model *say* "journal" later. No training signal ever aligned
the two: the cortex's 128-d state and its projection into the residual stream
were never trained against model behavior (llama.cpp had no gradients). The
emitted vectors were effectively random directions in activation space — and
random directions at high norm don't steer, they destroy.

**Evidence:** S2.1 (`2026-07-02_s2_1_minimal_loop.md`) — raw warm-state cvecs
at combined norm ~140 collapsed every generation into a repeated garbage
token; even clamped to combined norm 1.0 they still corrupted probes on
Qwen2.5-0.5B. S1.4 (`2026-07-01_s1_4_hf_layer_probe.md`) is the same lesson
from another angle: "the knowledge sits at layer L as an extractable vector"
did not survive contact with either architecture.

## What survives

1. **The thesis.** What died is its cheapest formalization: *changed dynamics
   = additive bias*. The mechanisms that can express conditional structure
   are KV entries (attention supplies the conditioning — S1.3 validated the
   channel) and weight deltas (multiplication makes them input-dependent: a
   LoRA fires only when the input excites its input directions —
   `research/18` pre-registers exactly this).
2. **The missing training loop.** Steering directions must be *learned
   against the model's behavior*, not accumulated from its perceptions. The
   Hebbian shortcut skipped that loop because the old substrate could not run
   it; the HF substrate can.
3. **A redeemable posture channel.** Once a working amplitude is calibrated
   for the current substrate, cvecs may legitimately carry what a constant
   direction can express — tone, caution, domain-prior — alongside a content
   channel that does the factual work. Posture in the residual stream,
   content in attention/weights: that division of labor is what the evidence
   supports.
