# Decoder parity DEV v1

Frozen September 13, 2026 under the user's language-interface request.
Manifest: `c9e35d5ca8b9eee57fc8a6740afc9b3bcc27cecee6fcc25198c4448a322b15ef`.
All old organ sources, outputs and instruments remain unchanged.

The DEV-v3 live comparison found scalar/native/batch discrepancies. The pinned
model generation configuration supplies repetition_penalty=1.1. Scalar decoding
uses raw argmax. The original batch function additionally supplies a fixed prompt
position_ids tensor. Transformers 5.0.0 slices those supplied positions during
continuation but extends only the attention mask, not that tensor.

The separate candidate `src/oczy/experiments/meta_cortex/generation_v2.py` uses an
explicit generation configuration and allows positions to be recomputed from
the expanding mask. It is not installed into the historical organ API.

This run isolates those factors in a 2 x 2 comparison:

| Condition | Repetition penalty | Continuation positions |
|---|---:|---|
| old_default | inherited 1.1 | original supplied tensor |
| old_neutral | 1.0 | original supplied tensor |
| dynamic_default | 1.1 | regenerated from attention mask |
| dynamic_neutral | 1.0 | regenerated from attention mask |

Raw scalar argmax is the comparison target. All four conditions see the same
frozen model, messages and bank. Temporary changes to the original API's
generation configuration are restored immediately and before final model-hash
verification. No weights, scorer, thresholds or old files are modified.

There are 32 cases: all 16 previous DEV-v3 calibration cases with oracle-selected
operations and a zero-width bank, plus all 16 DEV-v2 fresh probes without oracle
text using its already-saved seed-0 joint bank. That bank is copied and hashed
before execution. This is decode qualification on known DEV probes, not fresh
capability confirmation or feedback to the ongoing optimizer.

Native input-ID generation with neutral settings provides a further control on
all 16 zero-width cases. The candidate also runs all 32 cases in groups of four
variable-length prompts. Qualification requires exact decoded-text equality to
scalar responses for every candidate single-item and four-item batch response.
Targets are retained for separate accuracy audits; matching an incorrect scalar
answer still counts as decode parity, not successful task execution. This does
not establish bitwise token/logit equality or arbitrary batch/model equivalence.

Execution uses a separate two-thread CPU process, the exact pinned offline
runtime, no network, and a read-only source tree. Zero optimizer updates and no
meta-test. This run does not broaden the learning experiment's budget or bypass
its joint-capacity prerequisite for correction.

```bash
/home/nyanpasu/.local/state/oczy/diagnostics/2026-09-11-organ-identity/runtime/bin/python \
  scripts/probe_decoder_parity.py run \
  --root experiments/decoder-parity-dev-v1 \
  --output /home/nyanpasu/.local/state/oczy/diagnostics/2026-09-13-decoder-parity-dev-v1 \
  --model /home/nyanpasu/.local/state/oczy/diagnostics/2026-09-11-organ-identity/model \
  --provenance experiments_logs/artifacts/2026-09-13_capability_validation_v1/selected_runtime_provenance.json
```

Use a new output directory for reproduction.
