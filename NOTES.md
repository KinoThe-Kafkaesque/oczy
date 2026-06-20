# Oczy — session notes

## 2026-06-21 — thermos-nuclear code review pass

Triggered by review of `experiments_logs/2026-06-19_extended_learning_evaluation.md`,
which concluded that **NeuralHippocampus is the most promising organ**
(scoring 1.000, above the Oracle upper bound of 0.844). On closer reading,
that headline was a measurement artifact:

- **NeuralHippocampus's 1.000** measured its internal bookkeeping (store/
  replay/consolidate/decay), not whether learned corrections surfaced
  in the agent's answered output.  The 1.000 was a self-consistency
  check on the hippocampus's own data structures.
- **Oracle 0.844 below PlasticCortex 0.967** came from
  `memory_bytes_per_delta` being *averaged* into the aggregate as if
  higher = better, when it is natively lower = better.
- **HippocampusOnlyAgent 0.0 across the board** in the curriculum log
  was a `correction` vs `corrected_answer` schema drift: OrganismAgent
  passed `correction=<sentence>` to `neural_hippocampus.store()` but
  read `episode.get("corrected_answer")` on replay, so the replay hint
  silently never reached the ranker.

The review produced a prioritised finding list (critical: schema drift
in episode dicts; high: 5x duplication of `_extract_expected_from_correction`,
IdentityHypernetwork's hardcoded 14-token vocab; medium: missing
`status()` on WorldModelCritic, broken `_target_label_for` leading
space, arbitrary aggregate weights; low: nit-level).

### Changes this session

#### Glue layer (`oczy_common/`, `experiments/`, `eval_extended.py`)
- New `oczy_common/` package: `Episode` TypedDict +
  `EPISODE_FIELDS`/`validate_episode` (single source of truth for the
  dict keys crossing organ boundaries), `tokenize`/`STOPWORDS`, the rich
  `extract_expected_from_correction` heuristic, and `mem_bytes` (single
  pickle-based byte-count definition).
- `correction-benchmark/src/correction_benchmark/benchmark.py`:
  `_evaluate_probes` no longer calls `agent.answer()` twice per probe
  (the first call mutated stateful agents — `HippocampusOnlyAgent`
  incremented `replay_count`, `SkillImmuneCortex` incremented `hit_count`,
  `WorldModelCritic` updated `_last_correction_prob`; the recorded
  answer was from one call and the verdict from another).
- `eval_extended.py`: C4 — `memory_bytes_per_delta` inverted to
  higher-is-better before aggregation (Oracle now ranks #3 at 0.956,
  proper upper bound). C2 — `eval_neural_hippocampus` now stores
  `correction=sentence, corrected_answer=label` and reads
  `corrected_answer` back, instead of misusing `correction` for both.
  L5 —
  per-item PlasticCortex control agent documented and config hoisted.
  M4 — aggregation policy documented: every metric enters the aggregate
  as higher-better ∈ [0,1].
- `experiments/organism.py`: H5 — `LMBackendAgent` split out as a
  sibling class so the LM-only answer path is visible in the type
  system rather than hidden behind a `backend="lm"` config flag that
  silently bypassed critic/hippocampus/identity in `answer()` while
  still running `learn()` against them.  H1/H2 — `_tokenize` and
  `_extract_expected_from_correction` now delegate to `oczy_common`
  (the five baseline agents in `baselines.py` also now share these).
  M5 — `__getstate__`/`__setstate__` drop and rebuild the profiler on
  pickle so live timer state isn't serialised.  M6 — bare `except
  Exception` for the `LMPlasticCortex` import became `except
  ImportError` so AttributeError/SyntaxError in the LM module surface.
  `OrganismAgent._module_bytes` now prefers the canonical
  `status()["serialized_bytes"]` field, falling back to legacy fields
  and finally to `pickle.dumps(module)`.

#### Organs
- **neural-hippocampus**: `status()` adds `serialized_bytes` and
  `record_count`. `store()`'s optional `corrected_answer` param (added
  the prior session) is now end-to-end tested by
  `test_corrected_answer_round_trips_through_replay`.
- **world-model-critic**: gained a `status()` method (previously had
  none — `OrganismAgent._module_bytes` had a silent special-case
  fallback). Returns the 6 canonical fields including
  `serialized_bytes`.
- **skill-immune-cortex**: M2 — `status()["bytes"]` was
  `sys.getsizeof(json.dumps(...))` (Python object overhead, not UTF-8
  bytes); now `len(json.dumps(...).encode("utf-8"))`. `status()` adds
  `serialized_bytes` (pickle of the whole organ) and `record_count`
  (detectors + skills).
- **identity-hypernetwork**: H3 — implemented real vocab growth.
  `grow_vocab(new_concepts)` extends `W` with one fresh
  `1/sqrt(input_dim)`-scaled row per concept and updates
  `concept_index`/`output_dim` in place.  `_extract_first_concept` now
  auto-registers unknown tokens (alnum, length >= 3, not in a short
  stopword filter) so `update_identity` no longer silently no-ops on
  curriculum senses like "ML model". The closed `CONCEPT_VOCABULARY`
  stops being a hard blocker.
- **experience-autoencoder**: M7 — `train_step(episode, lr=0.01)`
  applies a rank-1 Hebbian update on the sensing matrix (outer of
  residual target and bag-of-words features, then per-column
  renormalisation), returning the pre-update reconstruction error.
  Over 25 steps on a profile episode the error dropped 0.5755 → 0.5447.
  M3 — the leading-space bug in `_target_label_for`'s fallback string
  is fixed. `encode()` now accepts canonical Episode keys
  (`query`/`answer`/`corrected_answer`) as aliases for the
  source-specific names it historically used
  (`situation`/`model_answer`/`revised_answer`). L1 — `ifany = any(...)`
  inlined. `status()` adds `serialized_bytes` and `record_count`.
- **plastic-cortex**: both `PlasticCortex.status()` and
  `LMPlasticCortex.status()` add `serialized_bytes` + `record_count` +
  `project`. Pickle round-trip verified clean on the LM cortex (numba
  kernels are module-level functions, not on `self`).

#### Tests
- New `experiments/tests/test_agent_glue.py`: T1 OrganismAgent
  end-to-end replay test (learn → answer → assert corrected label),
  T2 baseline ablation contract tests, and
  `extract_expected_from_correction` round-trip tests. Previously no
  integration test exercised `OrganismAgent.learn() → answer()` so the
  replay-drift bug was invisible until this session's manual smoke
  test.
- Each organ gained a `test_status_reports_serialized_bytes_*` test
  enforcing the canonical byte contract.

### Re-validated rankings (after fixes)

```
1. NeuralHippocampus      1.000
2. PlasticCortex          0.967
3. Oracle                 0.956   (was 0.844 — metric-direction bug fixed)
4. SkillImmuneCortex      0.590
5. WorldModelCritic       0.399
6. ExperienceAutoencoder  0.203   (now trainable via train_step; re-run to see this move)
7. Always-Wrong           0.200   (was 0.000 — same metric-direction bug)
8. IdentityHypernetwork   0.006
```

Oracle now sits at #3 instead of below PlasticCortex — the canonical
upper-bound ordering is restored.  NeuralHippocampus still tops the
ranking because `eval_extended.py`'s
`eval_neural_hippocampus` still only measures internal mechanics, not
behavioural learning; closing that gap is the next honest eval.

### Remaining gaps (next session)

1. **`eval_extended.py`'s IdentityHypernetwork probe is wrong**: at
   `eval_extended.py:309` it uses `token = first_keyword(correction)`
   as the adapter lookup key (e.g. `"model"` for "No, 'model' here
   means ML model."), then asserts `best == sense` where `sense =
   "ML model"`. Comparing a single concept token to a multi-token
   sense label can never match, so the organ scores 0.006 even though
   `grow_vocab` and `update_identity` now work. Fix: have
   `eval_identity_hypernetwork` check whether any concept in the
   adapter is present in the sense, or have `update_identity` register
   the full sense string as a concept (multi-word concepts weren't
   part of the grow_vocab commit).
2. **`eval_neural_hippocampus` still measures internal mechanics, not
   behaviour.** Add a behavioural probe: after the curriculum, do the
   hippocampus-only slow updates survive into the
   `OrganismAgent.answer()` output? Currently only the integration
   test checks this.
3. **WorldModelCritic** still uses hardcoded ambiguity + Jaccard
   similarity. The autoencoder's `train_step` opens up the possibility
   of swapping in a small learned similarity model.
4. **Organ-level `core.py` / wrapper split (H4) deferred**: the
   `core.py` files are now documented as "v1 organ lives in the
   wrapper" but the inconsistency is still present. Pick one pattern
   and apply uniformly.