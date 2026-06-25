# Oczy — Architecture Goals

The Plastic World Model Agent thesis (`experiments.txt`) calls for an agent
where memory becomes **changed dynamics**, not retrieved content. The current
codebase has the organ shapes stubbed but inverted: the LM is load-bearing on
the answer path while the cortex writes label strings. This file tracks the
three goals that un-invert the architecture and make the cortex the live
centre of the organism, with the LM demoted to a perception/articulation
organ on its edges.

The reference cortex contract lives in
`plastic-cortex/src/plastic_cortex/kv_cortex.py` (9/9 contract tests
passing as of 2026-06-23).

## Goal 1 — LM-side steering binding

The cortex emits per-layer steering vectors via `KVCortex.emit_cvec(layer_idx)`.
`LlamaCVecDriver` binds these into the LM's forward pass via
`llama_set_adapter_cvec`, which applies them as a per-layer residual bias.
This is sufficient to shift logits, but the available binding is a **residual
control vector**, not a reserved KV slot.

**Why it matters:** a control vector steers generation at the level of
*semantic posture*: framing, style, broad domain. It cannot reliably
override the LM's prior for a specific arbitrary token. Empirically, a small
projected cvec moves "profile" answers from "strategy" toward
"commercial"/"economic" but does not emit "vertical". A reserved KV slot lets
the cortex inject factual content into attention at a fixed sequence position
and force specific tokens.

**Status:** direct reserved KV-slot injection is **blocked** on the installed
`llama-cpp-python` binding. The package exposes `Llama.kv_self` opaquely and
does not provide a public API to write an arbitrary `(k, v)` tensor directly
into a chosen layer and position. C++ internal methods are exposed in the SO
but are mangled and tied to internal structures; calling them through ctypes
would require a binding fork or an upstream API addition.

**Proxy available today:** a fixed text prefix prepended to the prompt acts
as a reserved-position KV signal. The prefix tokens flow through the normal
forward pass and populate fixed KV positions, biasing generation toward the
encoded concept. In a 2026-06-25 probe this proxy achieved exact-token uptake
("vertical" appeared) where the residual cvec only achieved domain shift. The
prefix is therefore the practical fact-recall surface today; the cvec remains
the practical posture/framing surface.

**Done when (direct path):**
- A reserved KV slot per layer can be written and overwritten.
- The LM's next-token logits visibly shift when the slot is populated versus
  empty (a measurable behavioural test).
- Latency: <5 ms per cortex injection, so the loop can keep up with token
  streaming.

### What works today

`LlamaCVecDriver` provides two complementary surfaces:
1. Residual control vectors via `llama_set_adapter_cvec` for posture/framing.
2. An optional articulation prefix (`set_articulation_prefix`) prepended to
   every generate() prompt for exact fact recall.

The current codebase therefore uses **cvec for posture** and **prefix for
exact fact recall**, while Goal 1 remains a future architecture upgrade for
direct KV-slot writes.

### First-class reserved-position API (implemented)

`LlamaCVecDriver` now exposes `ReservedPosition` as a dataclass separate
from cvec steering.  The abstraction records provenance and optional
measured uptake so the organism can later learn which reserved positions
work:

```python
from oczy.lm import ReservedPosition

driver.set_reserved_position(
    ReservedPosition(
        text="In this codebase, profile means business vertical. ",
        source="knowledge_store",
    )
)
```

`LlamaCVecDriver` now provides two complementary surfaces:
1. Residual control vectors via `set_cvecs_per_layer` / `set_cvec_uniform`
   for posture/framing.
2. Reserved positions via `set_reserved_position` / `clear_reserved_position`
   for exact fact recall.

When a reserved position is set, the driver prepends `text` to every
`generate()` call.  If a cvec is also set, its scale must be attenuated
(`articulate_scale` < 0.01) under a prefix to prevent interference.

Long-term, this abstraction should accept a **prefilled KV cache chunk**
once the binding supports it, so the reserved position is not burned as
text tokens on every forward pass.

## Goal 2 — Hidden-state extraction at layer L

The cortex's warm path needs the LM's residual at a chosen depth L as
input to `observe(lm_hidden, correction_signal)`. The high-level
`Llama.create_chat_completion()` and `Llama.eval()` APIs hide internals,
so we need a path to read one layer's activations during a forward pass.

**Why it matters:** the cortex is currently either fed synthetic hidden
vectors (placeholder) or the embedding-layer output (insufficient depth to
capture reasoning). Real cortex metabolism needs the residual at a layer
that actually carries semantic intent — empirically layers in the middle
to upper third of a 28-layer transformer.

**Surface under investigation:**
- `llama-cpp-python` exposes `Llama._internal_state` style handles
  experimentally; needs verification.
- Alternative: run a "twin" eval on the prompt with a callback that
  captures intermediate activations.
- Heaviest option: fork `llama-cpp-python` and add a hookable eval
  path. Plan B only if the lighter paths fail.

**Done when:**
- `driver.peek_layer(prompt, layer_idx)` returns a `d_embd` array.
- The cortex, fed real layer-L hiddens, produces visibly different
  warm_state trajectories than when fed layer-0 hiddens.
- Pickling the cortex's `proj_hidden` after training on real
  hiddens shows non-trivial structure (not random).

## Goal 3 — Organ upgrades to tensor inputs

Today the five metabolism organs (`NeuralHippocampus`,
`WorldModelCritic`, `IdentityHypernetwork`, `SkillImmuneCortex`,
`ExperienceAutoencoder`) consume string features or parsed Episode
dicts. They need to consume tensor signals from the cortex:

| Organ | Today consumes | Should consume |
|---|---|---|
| NeuralHippocampus | parsed episode dicts, hash embeddings | hidden vectors from cortex; replay bank of tensors |
| WorldModelCritic | hand-built string features | `cortex.warm_cold_drift` as the prediction error |
| IdentityHypernetwork | `concept_scores` dict | cortex state deltas as the meta-state above warm |
| SkillImmuneCortex | keyword trigger strings | anti-direction KV-poison entries keyed on cortex state |
| ExperienceAutoencoder | already returns Δz | consume Δz via `cortex.train_step` (already wired on the cortex side) |

**Why it matters:** cortex metabolism without organ upgrades is a
no-op. The fast→replay→compression→slow loop must close through
these organs.

**Done when:**
- A correction through CortexAgent causes visible `cortex.cold_state`
  drift after `consolidate()`, NOT just `corrected_answer` string
  retrieval.
- Repeated corrections on the same concept produce compounding cold
  drift (not overwrite).
- The 6-stage organism curriculum's Stage 2 (scope control) becomes
  tractable: correction of one sense does not obliterate the other
  because they live in different cortex state regions, not the same
  label slot.

## Sequencing

Goal 1 unblocks everything: without KV injection the cortex can't
steer the LM, so whether the metabolism closes or not is invisible.
Goal 2 is required for the cortex's input side to see real LM
structure. Goal 3 makes the cortex metabolism actually mutate the
agent rather than just shift its own internal vector.

Strategic order: 1 → 2 → 3, but Goal 2's investigation can begin in
parallel once Goal 1's binding surface is understood.

## Working roadmap

- **Cortex/cvec:** changes dynamics, not facts.
- **Prefix/KV slot:** provides retrievable content.
- **Hippocampus/replay:** decides what content/state deserves consolidation.
- **Organs:** should become tensor consumers once the CortexAgent curriculum proves the loop.

## Non-goals (deferred until the loop closes)

- Differentiable plasticity (`alpha_ij` learned, not two scalars)
  — experiments.txt section 4.
- Energy/attractor basins as the cortex substrate — experiments.txt
  section 7.
- Implicit consolidation triggers (auto-fire on replay threshold).
- New curriculum stages beyond the six authored.
- LM cortex retraining (the 40K char-RNN side-quest).

## Identified sub-goal: meaningful cvec steering

Status as of 2026-06-24: **the placeholder path is closed.** See
`experiments_logs/2026-06-24_cortexagent_raw_hidden_steering.md` for
the full sweep. `KVCortex.steering_mode="raw_hidden"` is implemented:
`emit_cvec` broadcasts the L2-normalised `last_correction_hidden`
across all layers scaled by warm amplitude. At `articulate_scale=0.001`
(LFM2.5-1.2B-Instruct Q4_K_M) it produces clean on-manifold steering
toward the corrected sense; below `0.0005` it washes out, at `0.005+`
it falls into token-repetition garbage. The binary cliff of the
random projection (no-op until ~30, then crash) is collapsed to a
usable linear regime.

**Empirical finding behind this goal (2026-06-23 qualitative demo,
preserved for context):** the cortex at default random `proj_c`
produces noise-shape steering, not intent-shape cvec. At scale 0.5-16.0
the cortex had ZERO effect on greedy decoding; at scale 30+ it
crashed the LM into token-repetition garbage. There was no
"steering" regime, only a breakage threshold. `raw_hidden` is the
response to that finding.

Remaining work — make `proj_random` itself meaningful instead of
sidestepping it:

  (b) **SVD-initialised proj_c** (the follow-up): collect a small
      batch of real correction hiddens from the perception harness,
      SVD them, use the top-n right singular vectors as `proj_c`
      rows. Each correction's hidden vector then implicitly becomes
      one direction in the projector's range, so per-layer
      expressivity is preserved *and* the cvecs become
      semantically aligned. After this the default mode can return
      to `proj_random` (now meaningful) with `raw_hidden` kept as
      the coarse fallback.

Done when (updated status):
- ~~Articulate at SOME non-breaking scale produces output that visibly
  leans toward the corrected sense (not CJK garbage, not baseline).~~
  **Done 2026-06-24 via raw_hidden at scale 0.001.**
- The cvec norm at this scale is below the LM's residual-stream noise
  floor (so we're genuinely in the linear steering regime, not
  distribution-breaking). **Verified conceptually; the quantitative
  residual-noise-floor measurement is folded into Goal 1's
  logits-shift test.**