# Reproducing historical R20 DEV results locally

The R20 `meta_cortex/v2` organ hash includes the complete Hugging Face
configuration. Its `_name_or_path` field records the filesystem load path.
Identical model artifacts and package versions therefore hash differently
when loaded from the local HF cache and the Kaggle mount.

The September 11, 2026 diagnostic isolated this field: the local `2621e258…`
digest becomes the historical `a342431c…` digest when only the recorded
Kaggle path replaces `_name_or_path`. Restoring the field restores the local
digest. This was a diagnostic calculation, not authorization to rewrite a
checkpoint or bypass an identity check.

`scripts/reproduce_r20_dev.py` instead loads the verified files at their actual
historical path inside a private Bubblewrap mount namespace. It does not
modify the model configuration, checkpoint, scorer, manifest or expected hash.
The child has no network; source and model input mounts are read-only, with a
separate writable output directory. Nothing is submitted to Kaggle or Colab.

Prerequisites:

- Linux with `bwrap` and working unprivileged mount namespaces.
- A clean detached checkout at the source commit in the remote provenance.
- The exact Python and package versions recorded in the runtime manifest.
  For the corrected R20 v6 run: Python 3.12.13, Torch 2.10.0+cpu,
  TorchAO 0.17.0 (not 0.17.0+cpu), Transformers 5.0.0,
  Tokenizers 0.22.2, Safetensors 0.7.0; Accelerate 1.14.0 is also needed
  by the loader. The wrapper checks the complete existing runtime manifest.
- The matching local model snapshot, public calibration view, checkpoint,
  successful remote provenance and collected reference shard.

Run with the exact-runtime interpreter:

```bash
python scripts/reproduce_r20_dev.py \
  --provenance /absolute/path/remote_run_provenance.json \
  --source /absolute/path/clean-historical-checkout \
  --model-dir /absolute/path/local-hf-snapshot \
  --calibration-view /absolute/path/public/CALIBRATION_VIEW.json \
  --checkpoint /absolute/path/checkpoints/d0 \
  --output /absolute/path/new-reproduction-output \
  --task-index 0 --evaluation-seed-index 0 \
  --reference-shard /absolute/path/remote-shard.json
```

The task and seed must belong to the recorded remote DEV shard. The output
directory must be new. Model files are copied after hash verification so HF
snapshot symlinks work inside the historical mount and child output writes
cannot modify the model cache.

`reproduction.json` records the command, source/runtime identity, execution
outcome and (when a reference shard is supplied) exact comparisons of the
selected cell, no-update repeats, theta hashes and header identities. An
audit-field difference or duplicate record makes the comparison fail even if
the aggregate score is unchanged. Preserve failed attempts and logs.

This is a reproduction tool, not a new experiment or an R20 acceptance gate.
A matching all-zero cell leaves the scientific no-go in place. The historical
hash format remains unchanged; a future portable identity format would need
an explicit migration and must not silently relabel old checkpoints.

For focused diagnosis, `scripts/reproduce_r20_cell.py` calls the original
six-condition collector directly and compares every cell field against one
verified historical reference. Its output explicitly excludes the 20
no-update repeats and is not a calibration shard. Use
`OCZY_DIAGNOSTIC_SOURCE` to point it at the clean historical checkout's `src`
directory inside the same verified namespace. The full-shard path above can
be much slower because each repeat replays the complete online episode.

`scripts/probe_r20_articulation.py` is the next DEV diagnostic. It reads only
the public tuning view and compares unchanged no-context, teaching-context
and oracle-context outputs using the frozen exact scorer. It records
special-token-removal counterfactuals separately for adapter diagnosis; these
are not replacement scores and are not merged into historical calibration.

September 11 outcome: the focused cell matches all score and non-state-hash
fields, but four nonzero cortex-state hashes differ. Exact reproduction is
still unresolved. A previous full-shard attempt was stopped after 1,076.7 s
without producing a shard; the focused collector does not replace its omitted
20 no-update repeats. See the
[diagnostic log](../../experiments_logs/2026-09-11_campaign_r20_dev_output_path.md)
for this failure and the subsequent separately approved v3/v4 prompt checks.
