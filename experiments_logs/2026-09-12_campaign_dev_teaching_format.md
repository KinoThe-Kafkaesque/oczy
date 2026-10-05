# DEV teaching-format comparison and instrument coverage audit

**Date:** September 12, 2026. **Classification:** DEV DIAGNOSTIC ONLY.
The user instructed “get on the next blocker” after the next DEV teaching-format
comparison was proposed. The [v5 scope and authorization](../experiments/r23.5-serialization-dev/V5_CORRECTIVE_FACTS.md)
are recorded separately from the earlier v3/v4 approvals.

## Result: taught-answer readout is unblocked in this sample

The new `corrective_facts_context` condition copies only the existing public
corrections into a list of facts. It preserves the v4 system instruction,
query, target, exact scorer, generation settings and model. All original
conditions remain in the comparison.

| Condition | Contextual mapping | Transformation | Finite state | Total |
|---|---:|---:|---:|---:|
| No context | 0/3 | 0/2 | 0/2 | 0/7 |
| Original error/correction dialogue | 0/3 | 0/2 | 0/2 | 0/7 |
| Same corrections listed as facts | 0/3 | 2/2 | 2/2 | **4/7** |
| Oracle | 0/1 | 1/1 | 1/1 | 2/3 |

All 17 original v4 prompt hashes, outputs, targets and scores reproduce
exactly. The seven added generations differ only in teaching presentation.
The transformation outputs become exact `nurpnu` and `nurprho`; the two
finite-state outputs become exact `q2`. No output extraction or substring
credit is used.

Four of the seven queries are directly supported by the teaching examples,
as annotated before the run. All four now pass. The remaining three contextual
queries are untaught and remain in the full 4/7 table; their failures are not
filtered out. This is a successful text/retrieval presentation control, not
learned state, compression, or held-out rule generalization.

Execution took 35.47 seconds, exit 0, in the offline local historical model
namespace. The runtime manifest and ten model files were reverified before
execution. Organ hash remains `a342431c…` before and after; optimizer steps
are zero. Source is local uncommitted code on `e71268b`. No remote submission,
calibration payload access or meta-test access occurred.

Raw outputs and provenance: [2026-09-12_dev_corrective_facts.json](2026-09-12_dev_corrective_facts.json).
Frozen v5 proposal: `f87fee485939371596921498b9f6d0c227280c3a3d660d4092a5df67ea3b9cc6`.

## Deeper finding: some probes demand information absent from teaching

The selected contextual task teaches `coral/wix → left` and
`lavender/wix → rise`, but its three same-rule probes query `dax` in lavender,
coral and quartz. The generator draws these bindings independently; the two
observed `wix` facts do not identify the requested `dax` values.

A read-only audit covered all 105 public DEV tasks, not calibration or sealed
tasks. Original manifests and task bytes were verified. It found:

| Probe group | Public DEV finding |
|---|---|
| Contextual same-rule | **44/87** ask about untaught mappings; 37/75 training and 7/12 tuning |
| Contextual transfer | **30/70** ask about untaught mappings; 25/60 training and 5/10 tuning |
| Finite-state same-rule | All **70/70** directly taught |
| Transformation same-rule | All **70/70** directly taught |
| Finite-state transfer | All **59/59** query untaught, independently generated transitions |
| Contextual composition | All **35/35** use a second input absent from the declared mapping |
| Finite-state composition | All **35/35** require an independently generated action that is never taught |

Source causes are in `src/oczy/experiments/meta_cortex/taskgen.py`:

- `_ctxremap_same_rule_probes` accepts `taught_pairs` but ignores it, despite
  claiming to paraphrase taught mappings. `_ctxremap_transfer_probes` also
  ignores that set.
- Contextual composition passes an output token as the next symbol. The
  output and symbol vocabularies are disjoint, so the target silently falls
  back to the second context's last symbol. That fallback rule is not stated
  to the model.
- Finite-state transfer uses unseen edges from a randomly assigned graph.
  Composition expects `state action`, but learning events teach transitions
  only; the random action map is visible only in the oracle condition.

Transformation transfer is intentionally **not** classified as an arbitrary
missing lookup: its rule can potentially generalize to an unseen operand.
This audit does not prove all remaining tasks uniquely identifiable, and it
does not change scoring or re-adjudicate every historical experiment.

Detailed rows and counts: [2026-09-12_dev_teaching_coverage.json](2026-09-12_dev_teaching_coverage.json).
Audit implementation: `scripts/audit_dev_teaching_coverage.py`.

## Decision and next gate

The presentation blocker is resolved for this small taught-answer diagnostic.
An effect exists to investigate, but the existing R23.5 draft calls for
held-out probes, and no latent/soft-prompt serialization run has occurred.
Do not relabel this 4/7 retrieval result as cortex learning or serialization.

The next meaningful step is a separately frozen serialization pilot whose
held-out answers follow from the supplied examples. Before resuming full R20,
repair and version the task-support/composition defects and reconsider the
associated calibration design. Increasing compute cannot supply missing facts.
The original v2 measurements and no-go decision remain recorded; their scores
are not evidence of a clean language-organ learning ceiling.

V5 is not a repair to the task generator. Original tasks, manifests and
thresholds remain untouched. Exact historical cortex-state reproduction also
remains unresolved independently of the successful presentation control.

Input rendering also verifies on all 24 cases: every original message is
preserved, the system instruction remains first, and adapter token IDs equal
the tokenizer's direct chat-template input IDs. The maximum prompt is 263
tokens. The helper initially compared IDs to a `BatchEncoding` wrapper;
comparing its `input_ids` field fixes the audit helper, not the model path.

Validation: eight new tests passed for prompt/target preservation, all-public
event rendering, coverage classification, contradiction detection and refusal
to execute without matching approval. Ruff and whitespace checks pass.
