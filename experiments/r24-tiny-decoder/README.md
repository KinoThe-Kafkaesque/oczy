# R24 Tiny Shared Frozen Decoder — Proof of Concept

## Measurement status (2026-08-08)

The original `r24-tiny-decoder/v1` Kaggle numbers are **invalidated as
measurements**. V1 seeded after model construction, right-padded variable-length
queries without length-aware generation, omitted the oracle attention padding
mask, and admitted conflicting/undefined supervision rows. They remain useful
only as execution smoke tests.

The human-authorized `r24-tiny-decoder/v2` screen is frozen in
`v2_screen_plan.json`. V2 seeds independent RNG streams before construction,
keeps shared backbone initialization paired, evaluates equal query-length
buckets, masks oracle padding, removes invalid specificity/contextual-composition
rows with a fail-closed input-conflict audit, hashes rendered examples, saves and
reload-verifies weights, reports correct/total by family/kind, compares correct
state against swapped/random/zero controls, and retains retrieval baselines.
The viewed seed-123 learnability ladder is diagnostic only; the screen uses a
fresh tuning catalog, and confirmation will use a separately frozen catalog and
five seeds.

```bash
# Learnability ladder
uv run python -m oczy.experiments.r24_tiny_decoder \
  --protocol-version v2 --diagnostic-ladder --output /tmp/r24-v2-ladder

# One frozen screen case
uv run python -m oczy.experiments.r24_tiny_decoder.suite_v2 \
  --case base --output /tmp/r24-v2-base
```

Implements proposal: one shared frozen byte-level decoder conditioned by cortex state r[64],
not per-unit decoders.

## Architecture
- **Vocab 260**: 256 byte values + PAD/BOS/EOS/UNK (covers task charset 28, no OOV)
- **Tiny Transformer**: 2-4 layers, 2 heads, d_model 64/128, max_len 512, dropout 0.1
- **Conditioning**: FiLM `x = γ(r)⊙x + β(r)` vs additive `x + W·r` (R25 ablation), shallow vs deep (every layer)
- **Loss**: exact autoregressive byte CE, teacher-forced shift `logits[:, Tq-1:Tq+Ta-1] vs answer`
- **Frozen artifact**: `parameter_hash` SHA-256 over sorted params, `freeze()` asserts hash stability

Params: 144k (2L/64d) – 920k (4L/128d) vs Qwen 0.5B (500M) → 133× forward speedup (0.4s→0.003s CPU).

## Phases per proposal

### Phase A — organ pretraining (oracle supervision)
```
complete rule text (mapping table) -> TextOracleEncoder -> z*[64]
query bytes + z* -> TinySharedDecoder -> answer bytes   byte CE
```
- Same `z*` serves many queries per rule (`rule_fingerprint` groups probes)
- `TextOracleEncoder`: 2-layer Transformer byte-embedded oracle_text → mean-pool → tanh → 64, trainable, generalizes to DEV rules (unlike fixed hash or PerRuleEmbedding which memorize)
- Split firewall: `build_dev_catalog` train/meta_validation split by complete rule fingerprints (no paraphrase leakage, no meta-test)
- Same-query/different-rule pairs: 77 queries (e.g. `wix demands what token?`) have ≥2 answers, forcing use of `r`
- Train 30/10 tasks/family → 479 train probes, 159 val probes, avg answer len 5.7, max 23

Controls: `oracle DEV acc` vs `query-only (r=0) DEV acc`. Delta >0.02 required (R24 success).

Current POC: `d64/2L/film` 800 steps lr 3e-3 → oracle 13.2% query-only 10.6% delta 2.5% (passes R24 ≥2% on that seed); other seeds/hypers give 0–5% — hyperparam search needed, but mechanism exists.

### Phase B — freeze artifact
```json
{ "config": {...}, "weight_hash": "b134...", "corpus_hash": "...", "split_hash": "...", "seed":..., "optimizer":{...}, "oracle_dev_accuracy":..., "query_only":... }
```
Weights frozen, `requires_grad=False`, hash bit-identical before/after freeze.

### Phase C — cortex integration
```
r = Rθ(F,S,query)   // MetaCortex read  [B,64]
decoder(query_bytes, r) -> answer   // gradients flow through FROZEN decoder into cortex
```
- `CortexDecoderBridge` wraps `MetaCortex` (write/consolidate/read) + frozen `TinySharedDecoder`
- During developmental training, optimizer steps only `theta_cortex` (writer/reader/consolidator), not decoder
- At evaluation, only `F/S ∈ R^{64×64}` may change — matches `research/20:64-77` boundary, no optimizer.

Causal controls (proposal 1–9): zero cortex, random cortex, trained, shuffled feedback, zeroed state, swapped state, same-query/different-state, byte-matched retrieval, oracle upper bound. Decisive: `D(q,r_correct) ≠ D(q,r_zero/swapped)` with decoder hash identical.

Cost: “one afternoon, CPU or single small GPU” per R24; INT8 Qwen campaign exceeded 12h at 15 tasks/family (log 959e114:49-80), tiny decoder fits easily.

## Files
- `src/oczy/experiments/r24_tiny_decoder/vocab.py` — byte vocab
- `src/oczy/experiments/r24_tiny_decoder/decoder.py` — `TinySharedDecoder`
- `src/oczy/experiments/r24_tiny_decoder/oracle.py` — `TextOracleEncoder` / `PerRuleOracleEncoder`
- `src/oczy/experiments/r24_tiny_decoder/pretrain.py` — Phase A loop, corpus, `train_phase_A()`
- `src/oczy/experiments/r24_tiny_decoder/cortex_bridge.py` — Phase C bridge
- `src/oczy/experiments/r24_tiny_decoder/controls.py` — 9 controls enumeration

## Usage
```bash
.venv/bin/python -c "from oczy.experiments.r24_tiny_decoder.pretrain import train_phase_A; train_phase_A(train_per_family=30, val_per_family=10, d_model=64, n_layers=2, conditioning='film', steps=1000)"
.venv/bin/python -c "from oczy.experiments.r24_tiny_decoder.decoder import *; m=TinySharedDecoder(TinyDecoderConfig()); print(m.parameter_hash()[:8])"
```

## Next steps to authorize R24
1. Human sign-off on this POC (new pre-registered version, not R20 v2)
2. Sweep `d_model {64,128} × layers {2,4} × conditioning {film,additive} × deep_film {F,T}` on 30/10, 1500 steps, 5 seeds
3. Add retrieval baseline (`C8` byte-matched) and meta-trained vs random cortex (R26)
4. Only after toy accepts, consider `meta_cortex/v3` with tiny decoder
