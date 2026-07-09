# 2026-06-25 — SVD-init'd proj_c: direction survives cold boot

Session carried the "meaningful cvec steering" sub-goal past the
2026-06-24 raw_hidden finish line. raw_hidden produced clean on-manifold
steering at `articulate_scale=0.001`, but its steering DIRECTION lived
in `last_correction_hidden` — a warm-side field that
`CortexAgent.save()` deliberately does not persist. On cold boot the
cvector collapsed to zeros and the conversation demo's `post-reload`
turn was byte-identical to `baseline`. This session closes that gap by
moving the direction into `proj_c`, which IS persisted.

## Two bugs found and fixed along the way

Before the SVD work could matter, two field-drop / shadowing bugs in
`CortexAgent` had to be cleared:

1. **`steering_mode` dropped by config reshape.**
   `CortexAgent.__init__` rebuilt `KVCortexConfig` by hand with nine
   fields, silently dropping `steering_mode`. The demo could request
   `raw_hidden` all it liked; the cortex always ran in `proj_random`.
   Surfaced as `选选选…` CJK garbage (the proj_random crash signature at
   scale 30) where the 2026-06-24 log claimed clean steering. Fix:
   `dataclasses.replace(ccfg, d_embd=..., n_layers=...)` — preserve
   every caller-set field; only override the shape mirrors.

2. **`articulate_scale` shadowed by instance attribute.**
   `articulate()` reads `self.config.articulate_scale` (the dataclass
   field). Setting `agent.articulate_scale = X` after construction
   only creates a shadowing instance attribute that `articulate()`
   never sees. Masked for several iterations because a debug print
   reading `reloaded.articulate_scale` returned `0.3` correctly while
   the actual code path used `self.config.articulate_scale = 0.03`.
   Fix: set the scale in the `CortexAgentConfig` passed to
   `CortexAgent.load(...)`; never via post-construction assignment.

Both are saved as feedback memories; both are the kind of silent
silent-failure that would have masked the SVD work if not cleared first.

## What landed in the code

`KVCortex.init_proj_c_from_svd(hiddens)` — new in-place method on
`plastic-cortex/src/plastic_cortex/kv_cortex.py`:

- Centres `hiddens` (shape `(N, d_embd)`), runs SVD, takes top-`d_cortex`
  right singular vectors as the projector's basis.
- Broadcasts the SAME slab across all `n_layers` (per-layer Gaussian
  perturbation would reintroduce the off-manifold cliff that motivated
  raw_hidden in the first place).
- Each column is a unit-norm singular vector scaled by `1/sqrt(d_cortex)`,
  matching `proj_random`'s bound convention so `emit_cvec` magnitudes
  are comparable across modes.
- Refuses `N < d_cortex` (would yield degenerate singular vectors).
- Marks `_dirty` so the next `emit_cvec` regenerates payloads.

No new `KVCortexConfig` field. The SVD path needs external hiddens data;
an on-config flag would force sourcing that data via env/global.
`KVCortexConfig` is part of the `CortexAgent` pickle, so keeping it
data-only preserves serialisability. Callers wire data by calling
`agent.cortex.init_proj_c_from_svd(np.load(...))` after boot.

## Collection harness

`experiments/lm_perception/collect_correction_hiddens.py` produces
`reports/correction_hiddens.npy` of shape `(72, 2048)`. Twelve canonical
correction episodes from `correction-benchmark/dataset.py` × six
paraphrase templates. Mirrors `cortex_agent.py:197` exactly:
`CortexAgent` + `CVecDriverConfig(n_ctx=512, embedding=True)` +
`peek_embedding(utterance, last_token_only=False)`. Distribution match
to `perceive()` is automatic by construction. Top singular values:
`[179, 153, 133, 120, 112, 109, 100, 99]` — non-degenerate spread.

Final-layer-only substrate is the smallest-fix path; Goal 2 layer-L
extraction can refine the basis later without changing
`init_proj_c_from_svd`'s contract.

## Demo

`experiments/cortex_conversation_demo.py` switched both config sites
from `raw_hidden` back to default `proj_random`, with SVD-init at boot.

Scale sweep on the live steered turn (post-correction, warm_norm ~6.83):

| `articulate_scale` | LM output | diagnosis |
|---|---|---|
| 0.001 – 0.01 | byte-identical to baseline | below noise floor |
| 0.03 | `"**Answer:** In this product, 'profile' refers to a detailed description of the product's"` | clean steering, on-manifold |
| 0.1 | `'Celle (C) : Cellen - C : Cellen…'` | off-manifold degenerate |
| 0.3+ | `'flora flora flora…'` / `'redesign redesign…'` | token-repetition crash |

Reload sweep (warm_state dampened by consolidate()'s 5% EMA, ~0.34):

| reload `articulate_scale` | LM output | vs baseline |
|---|---|---|
| 0.1 | byte-identical options list | no effect |
| 0.3 | `"**Answer:** In this product, 'profile' refers to a detailed description of the product's"` | **byte-identical to live steered** |
| 1.0+ | repetition crash | off-manifold |

The ~10× ratio between the steered scale (0.03) and the reload scale
(0.3) is the 20× consolidation dampening divided by the noise floor
width — it is per-host-residual-norm dependent and must be re-swept
if the LM or its quant changes.

## What is verified

- `init_proj_c_from_svd` lands `proj_c` exactly on the top-`d_cortex`
  right singular vectors of the centered hiddens, broadcast identically
  across all layers. Asserted in `test_svd_init_proj_c_structure`
  (`plastic-cortex/tests/test_kv_cortex.py`): all slabs identical;
  `proj_c[0].T @ proj_c[0]` matches `Vt[:d_cortex].T @ Vt[:d_cortex] /
  d_cortex` within atol=1e-5; column norms are `1/sqrt(d_cortex)`;
  pickle round-trip byte-for-byte. 12/12 cortex tests pass.
- SVD-init'd `proj_c` survives `CortexAgent.save()` / `load()`:
  `proj_c` is restored at `cortex_agent.py:454` from the pickle
  unchanged. The reload turn is NOT re-injected with SVD-init, so any
  steering it produces is by-construction proof of persistence.
- End-to-end demo: `post-reload` is byte-identical to `steered`, both
  producing `"**Answer:** In this product, 'profile' refers to a
  detailed description of the product's"`. `baseline != steered` True,
  `steered != post-reload` False, `baseline != post-reload` True.
  Verdict: *"cortex steering persists across consolidate + save/load."*

This **closes** the "meaningful cvec steering" sub-goal's remaining
follow-up — `proj_random` itself is now meaningful (SVD-seeded), and
`raw_hidden` is retained as the documented coarse fallback. The
default can stay in `proj_random` permanently; raw_hidden is no longer
load-bearing for the demo.

## What is NOT established — and an important falsification

Hypothesis tested: *"if corrected enough times, the correction becomes
new knowledge the LM can recall."*

Probe: taught two fabricated facts the LM cannot know from pretraining
(`zarnox_flux() returns 7331`, `plenvik_brim() returns 4096`) as
corrections, 20 cycles each with `metabolize()` + `consolidate()` between.
Then probed on a clean prompt at gentle (0.001–0.03) and breaking (0.1+)
scales.

| What | Result |
|---|---|
| `cold_state` norm after 20 cycles | `0.34 → 3.6` (10× accumulation — corrections DO compound) |
| Taught token `7331` in any output? | No, at no scale |
| Taught token `4096` in any output? | No, at no scale |
| Gentle-scale output | pretrained refusal ("not a recognized function") |
| Medium-scale (0.03) output | confidently hallucinated context ("commonly used in GitHub Actions") — wrong but fluent |
| Breaking-scale (0.1+) output | token-repetition crash |

**Finding:** the cortex accumulates state (your mechanism intuition is
right) but the state is a posture bias, not knowledge the LM can recall.
The cortex's state is a single `(d_cortex=64)` direction projected
through `proj_c` into a `(d_embd=2048,)` residual bias broadcast across
all 16 layers. That bias can reweight existing plausible continuations
(this is why the demo worked — "profile → business vertical" is a real
ambiguity in the LM's token distribution). It cannot manufacture tokens
the LM has no pretrained prior on, nor store `(question → answer)`
tuples — there's nothing to retrieve.

This is experiments.txt's "memory becomes changed dynamics, not
retrieved content" thesis taken literally. It works for dynamics
(steering, posture, framing, word-sense disambiguation). It breaks for
content recall — the right answer has to already live, weakly-biased-toward,
inside the LM's pretrained weight distribution.

What would actually deliver "correct enough times → new knowledge":
Goal 1's reserved-KV-slot writes — real `(k, v)` pairs into an attention
KV slot so the LM's own attention can retrieve them. That is
retrieval-as-memory, structurally distinct from the cortex's
steering-as-dynamics. Saved as project memory.

## Where this leaves the goal graph

- **Closed:** "meaningful cvec steering" in full. `proj_random` is
  SVD-seeded and persisting; `raw_hidden` is the coarse fallback.
- **Open:** *differentiable plasticity* on `proj_c` (currently no
  post-init training path; `train_step` only updates `proj_hidden`).
  Per-layer SVD bases (different slab per layer) is also deferred —
  per-layer noise reintroduces the cliff.
- **Newly urgent:** Goal 1 (reserved-KV-slot writes) is now the only
  path to actual recall of taught facts. The cortex steering work
  proved that steering alone cannot manufacture tokens; retrieval
  into attention can. That is the next milestone.

## Files touched this session

- `plastic-cortex/src/plastic_cortex/kv_cortex.py` — new
  `init_proj_c_from_svd` method.
- `plastic-cortex/tests/test_kv_cortex.py` —
  `test_svd_init_proj_c_structure`,
  `test_svd_init_rejects_undersized_hiddens`. 12/12 pass.
- `experiments/cortex_agent.py` — `dataclasses.replace` for config
  reshape (field-drop fix).
- `experiments/cortex_conversation_demo.py` — switched to
  SVD-init'd `proj_random` at scale 0.03 (steered) / 0.3 (reload).
- `experiments/lm_perception/collect_correction_hiddens.py` — new
  harness; outputs `reports/correction_hiddens.npy` (72×2048) +
  `correction_hiddens_meta.json`.
- `experiments/lm_perception/reports/correction_hiddens.npy` —
  generated dataset (committed).