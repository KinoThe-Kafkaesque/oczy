# Scoped diversity DEV v3

Authorized by the user's September 13 request to pursue the next objective,
check previous capabilities for regression, and package one experience.
Manifest: `e339fd54cedba4a039af0e98c2008b5934b781aff720f5da36ab2242c72c70b6`.
All execution sources, data, numeric controls and choices are frozen before
model calls. Historical instruments remain untouched; this is local DEV only.

The sole learner change is teaching-word diversity. The earlier control used
oak/pine/elm under amber, cobalt and silver for every update. The new learner
rotates three nine-example groups: those words, birch/cedar/ash, and
beech/maple/willow. Each group crosses its words with the same three contexts.
Forty-eight updates expose each context/word pair sixteen times. There are still
nine example losses per update, using the same Adam settings, learning rate .03,
gradient clip 1, mean CE including EOS, concise prompt and shared 8 x 896 FP32
bank. This fixes update/example counts, not exact token counts or FLOPs.
Prompt/target token-length distributions and elapsed time are recorded.

Seeds 0/1/2 start from the exact old initial bank hashes. The first update uses
the old examples and checks its pre-update loss against the archived control.
The old final control banks are reused directly; they are not retrained or
selected. All seeds and final states are reported. No adaptive training,
held-out tuning, checkpoint selection or automatic budget increase is allowed.

Both conditions are evaluated on identical new words berry/cherry/papaya/orange
under amber (append vek), cobalt (append mip), taught-neutral silver and untaught
quartz (copy). The previous lime/melon/grape/peach cohort remains a separate
regression set. Initial banks and a zeroed bank are controls. References retain
no-context, direct-operation oracle, complete rule table, text and conversation
example retrieval, and independently decompressed/regenerated zlib retrieval.
Both teaching sets appear in retrieval comparisons. Logit-bias/rerank are not
implemented and are explicitly reported as not run. Numeric and text storage
are counted separately; oracle-selected operations are not learned selection.

The primary result is paired scoped generalization and neutral retention.
Admission to correction requires all 27 teaching fits and all 16 new-word
responses correct for every diversity seed. On admission only, the already
specified 24-update unprotected/proximal correction comparison runs with the
same three new amber examples, no old-example replay, and coefficient 10.
Otherwise both correction phases remain unstarted. A gate failure is a result,
not evidence of mathematical impossibility or a refutation of the unrun penalty.

Each phase runs in an offline four-thread CPU process with the pinned INT8
Qwen organ and unchanged exact scorer. The training worker cannot access the
registered probe/oracle/control directories or the data-builder source. The
final bank alone is released to a new evaluation process. These are audited
worker boundaries, not protection against arbitrary malicious host code.
The scalar decoder remains fixed for the one-variable learner comparison.
The separately qualified batch helper can be exercised by the package and
regression checks without replacing historical source files.

Seven curriculum checks and twenty-one eval-guard checks pass before launch.
Runtime/model identities, gradients, numeric transfers, coverage, scores and
prompt hashes are audited. Subsequent regressions distinguish reproduction of
previous failures from successful capabilities and measure effects of the new
state separately. The package is an inspectable DEV workbench, not a claim of
continual-learning acceptance or an R20/meta-test result.

```bash
/home/nyanpasu/.local/state/oczy/diagnostics/2026-09-11-organ-identity/runtime/bin/python \
  -m scripts.diversity_dev.run \
  --root experiments/scoped-diversity-dev-v3 \
  --output /home/nyanpasu/.local/state/oczy/diagnostics/2026-09-13-scoped-diversity-dev-v3 \
  --model /home/nyanpasu/.local/state/oczy/diagnostics/2026-09-11-organ-identity/model \
  --provenance experiments_logs/artifacts/2026-09-13_capability_validation_v1/selected_runtime_provenance.json
```

Use a new output directory for reproduction. No remote jobs, old loop restarts,
meta-test, commits, pushes or production operations are part of this run.
