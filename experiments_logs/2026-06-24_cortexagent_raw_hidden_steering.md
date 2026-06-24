# 2026-06-24 — CortexAgent raw_hidden steering: the binary cliff, mapped

Session `c247c527` (carried over from 2026-06-23 evening) attempted to
close the "meaningful cvec steering" sub-goal identified at the end of
the 2026-06-23 log: implement path (b) — a `raw_hidden` emission mode
on `KVCortex` that anchors the cvec on the actual last-correction
hidden vector instead of a random projection, then sweep amplitude to
find a non-breaking steering regime.

A note on the session itself: the hosted model produced several long
runs of off-task CJK novel output interleaved with the real work
(`/compact` did not retract it cleanly), so the conversation was
"corrupted" in the sense that the live transcript is unusable as a
record. This log is reconstructed from the actual code changes and
the final English summary that did land. The findings are trustworthy
because they are reproducible from `experiments/cortex_agent.py` and
`plastic-cortex/src/plastic_cortex/kv_cortex.py`.

## What landed in the code

`KVCortex` gained a steering-mode switch and the raw-hidden path it
gates.

- `KVCortexConfig.steering_mode`: `"proj_random"` (default,
  backwards-compatible) or `"raw_hidden"` (the new semantically-aligned
  anchor).
- `KVCortex.last_correction_hidden`: a `d_embd` vector captured inside
  `observe()` whenever `correction_signal > 0.5`. It stores the LM
  residual that was present when the last genuine correction fired —
  i.e. the cortex now remembers *which direction* the last correction
  pushed.
- `_recompute_payloads()` branches on `steering_mode`:
  - `proj_random` keeps `proj_c @ warm_state` (random projection,
    per-layer expressive but semantically arbitrary).
  - `raw_hidden` broadcasts `unit_h * amp` across all layers, where
    `unit_h` is the L2-normalised `last_correction_hidden` and `amp`
    is the cortex warm amplitude. One vector, no per-layer
  expressivity, but the direction is the *correction's own* direction.

`CortexAgentConfig.articulate_scale` default moved to `0.001`. The
docstring records the empirical sweep that produced it:

```
# on LFM2.5-1.2B-Instruct Q4_K_M with steering_mode="raw_hidden":
#   scale < 0.0005  -> no effect on greedy decoding
#   scale 0.001     -> clean steering, output shifts toward correction
#   scale >= 0.005  -> off-manifold, falls into token-repetition garbage
```

The scale is explicitly flagged as per-host-LM-residual-norm
dependent: re-sweep if the LM or its quant changes.

## The sweep — the binary cliff, measured

On LFM2.5-1.2B-Instruct (Q4_K_M) with `steering_mode="raw_hidden"`,
correcting the cortex toward a target sense and then articulating:

| `articulate_scale` | LM output | diagnosis |
|---|---|---|
| 0.5 – 16.0 (`proj_random`, prior default path) | byte-identical to baseline | cvec washes out — no steering regime exists |
| 30.0 (`proj_random`) | CJK garbage (`选选选…`) | breakage threshold crossed, off-manifold |
| 0.001 (`raw_hidden`) | structurally different, *on-manifold* | **clean steering** — LM answers "what does profile mean?" instead of listing options |

The shape of this curve is the real finding: there is no continuous
"steering amplitude" knob on the random projection. The cortex's
intent either washes out entirely (because a random direction times
any sub-breakage amplitude is below the residual stream's noise
floor) or breaks the LM (because the amplitude needed to flip greedy
decoding through a random direction is large enough to leave the
data manifold). Between those two regimes there is no usable middle.

`raw_hidden` collapses that cliff. Because the cvec direction is now
the correction's own hidden direction, a genuinely small amplitude
(`1e-3`, below the residual noise floor) is enough to nudge greedy
decoding toward the corrected sense *without* leaving the manifold.
The same amplitude through `proj_random` does nothing visible, and
the amplitude the random path needs (~30) is destructive.

## What is verified

- Cortex mutation + persistence: `consolidate()`, `save()`, `load()`
  round-trip cleanly. The mutated `cold_state` survives across
  driver reloads. (This was already true at end of 2026-06-23; the
  raw_hidden change did not regress it.)
- `last_correction_hidden` is captured and unit-normalised correctly;
  `_recompute_payloads` produces a finite, on-distribution cvec.
- The articulation path end-to-end: `perceive → correct →
  consolidate → articulate` visibly shifts greedy output at
  `scale=0.001` and does not shift it at `scale < 0.0005`. The
  surface is closed.

## What is NOT established

- **Per-layer expressivity is sacrificed.** `raw_hidden` broadcasts
  one direction across all 16 attention layers. If different layers
  need different intents (the original reason `proj_c` had
  per-layer rows), this mode cannot express that.
- **Single correction direction only.** Only the *last* correction's
  hidden is kept. The cortex's accumulated warm_state still drives
  `amp`, but the directional content is whichever correction fired
  most recently.
- **Scale is host-specific.** `0.001` is a property of
  LFM2.5-1.2B-Instruct Q4_K_M's residual stream norm, not a
  universal constant. Switching models means re-sweeping.
- **No quantitative behavioural test yet.** "Output shifts toward the
  corrected sense" is the qualitative demo from the existing
  perception-layer harness. Goal 1's "logits visibly shift" success
  criterion is the next concrete check.

## Where this leaves the goal graph

This **closes** the "Newly identified sub-goal: meaningful cvec
steering" at the bottom of `GOALS.md` — the placeholder raw-hidden
path (option (a) in the goals file, option (b) in the prior log —
same idea, two labellings) is implemented and demonstrated to
produce a real steering regime.

The remaining work, where `proj_random` itself becomes meaningful
rather than being sidestepped, is the SVD-initialisation path:

  collect a small batch of real correction hiddens from the
  perception harness, run an SVD, initialise `proj_c`'s rows from
  the top-n right singular vectors. Each correction's hidden then
  implicitly becomes one direction in the projector's range, so the
  cortex keeps per-layer expressivity *and* gets semantically
  aligned cvecs. After that the default mode can return to
  `proj_random` (now meaningful) and `raw_hidden` stays as the
  coarse fallback.

That is the natural Goal 1 follow-up: it is the "make the cortex
meaningful" predecessor to Goal 1's reserved-KV-slot injection test,
because injecting a noise-shaped cvec into a reserved slot would
just shift the cliff rather than remove it.

## Files touched this session

- `plastic-cortex/src/plastic_cortex/kv_cortex.py` —
  `steering_mode`, `last_correction_hidden`, `_recompute_payloads`
  branch.
- `experiments/cortex_agent.py` — `articulate_scale=0.001` default
  and the empirical sweep docstring; the CortexAgent class that
  owns one driver + one cortex + the five organs (un-inverted
  ownership direction).
- `experiments/tests/test_kv_cortex.py` — raw_hidden emission
  covered.
- `experiments/cortex_conversation_demo.py` — qualitative harness
  that produces the sweep table above.