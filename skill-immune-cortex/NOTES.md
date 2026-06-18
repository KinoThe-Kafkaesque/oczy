# Skill Immune Cortex —prototype notes

## What is implemented

- `MistakeDetector`: a small, bounded failure signature with trigger keywords and a forced distinction/response.
- `SkillImmuneCortex`:
  - `add_detector()` turns a past correction into a detector and tracks mistake classes.
  - `check()` returns active immune responses when a query or proposed answer matches detector or skill triggers.
  - `merge_detectors()` collapses same-class detectors into broader detectors.
  - `status()` reports detector count, merged detector count, skill count, class counts, and approximate size.
- `Skill`: reusable competence object compiled when the same mistake class reaches a configurable repetition threshold.

## Limitations of this prototype

- **Trigger matching is naive**: lowercase substring search only, no stemming, no semantic similarity, no regex support beyond what tokens happen to contain.
- **Trigger extraction is heuristic**: stopword filtering and minimum token length may drop meaningful short tokens; if nothing survives, it falls back to raw words.
- **Merging is by class label only**: two detectors with semantically similar content but different `mistake_class` values are never merged.
- **Skills are not learned weights**: they are explicit trigger/policy objects, not distilled into adapter weights or trainable parameters.
- **No scope or context sensitivity**: a trigger fires everywhere it appears, regardless of domain or user intent, so over-generalization is possible.
- **No persistence or replay**: detectors live only in memory; there is no storage, consolidation, or immunity against old mistakes over time.
- **No contradiction handling**: if corrections later contradict each other, the cortex keeps both detectors active.
- **Size metric is approximate**: `status()["bytes"]` uses the serialized JSON size, not a true memory-trace accounting.

## Next steps

1. **Semantic triggers**: replace substring matching with embeddings or a small matching model so related phrasing activates detectors even with different wording.
2. **Context-scoped activation**: scope detectors by domain, project, or user so a correction learned in one context does not over-generalize.
3. **Confidence / critic gate**: add a small critic that decides whether a detector should fire for a given answer, rather than firing purely on trigger presence.
4. **Contradiction detection**: track when new corrections conflict with existing detectors/skills and flag them for review or decay.
5. **Distillation into skills**: once a skill is used successfully, compress its policy into a compact adapter or weight delta rather than an explicit text rule.
6. **Persistence and replay**: serialize detectors, log activations, and periodically replay old failures to test that the immunity still holds after consolidations.
7. **Evaluation harness**: measure correction-to-competence metrics from `experiments.txt` section 14, especially `behavior_delta_per_byte_of_persistent_memory`.
