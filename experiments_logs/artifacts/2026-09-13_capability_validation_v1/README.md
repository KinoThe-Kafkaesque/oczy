# Six-capability DEV v1 reproducibility archive

The local run completed September 13, 2026. The manifest was frozen before
execution. This archive preserves all four phase outputs, trajectories, twelve
numeric checkpoints, serialized text/compressed text, actual tool transcripts
and filesystem snapshots, execution commands, runtime manifest, input data,
bound execution sources, tests, and the final analytical report.

Instrument manifest: `74d6614fc332c832761c208e63e753dd85f437b850ebfaeeac196faeccdf53ce`.
Source base: `e71268bc9cb01839995e5b3db69d0ad43a9ef04d`; overlay `source/`
on that repository revision, and place `instrument/` at
`experiments/capability-validation-v1`. This was a local uncommitted diagnostic,
not a remote clean-commit campaign. No legacy eval or meta-test was modified.

Use the command in the instrument README with a new output directory. The
included `selected_runtime_provenance.json` can be supplied to `--provenance`;
it contains exactly the original model path and pinned runtime manifest needed
by the runner, excluding unrelated account/job metadata. Model weights and the
Python runtime are shared external dependencies and are not duplicated here.
They must match `run/runtime_manifest.json`; the runner verifies all model
artifact hashes and exact package versions before execution.

The analytical Markdown report is copied verbatim from `experiments_logs/`,
so its relative links are intended for its canonical repository location.
Full JSON outputs and all referenced runtime artifacts are also included here.
`VERIFICATION.json` records independent checks and action component counts.
`SHA256SUMS` covers every file in this archive except itself.

No-capability-promotion result: new teaching fits, but scope and old knowledge
are not preserved. Distinguish actual tool execution from final-answer format,
and ordinary lossless text compression from learned neural compression.
