# 28 — User-pattern generalization boundary

**DRAFT — proposed 2026-07-26 from external review**
(`chat-export-1785143922754.json`, Qwen3.8-Max-Preview). Not yet
human-approved pre-registration. Agents MUST NOT run this experiment until
it is human-authorized.

## Source and evidence boundary

The external review flagged that the Oczy thesis says "unseen rules" but has
never defined what makes a rule unseen. Under the "single-patterned users"
insight, the variety is *across* users, not *within* a user. The right
generalization gradient is over **user-pattern space**, not over structural
task complexity. This entry replaces the vague "unseen rules" language with a
measurable generalization profile.

## Problem

A single binary "unseen rule" test is uninterpretable. If the system passes,
you do not know how far from the training distribution the test rule was. If
it fails, you do not know whether it failed because the rule was slightly
different or structurally alien. This entry forces a precise definition of
"unseen" by measuring generalization across a gradient of user-pattern
distance from the meta-training prior.

## Hypothesis

**H-GENERALIZATION-GRADIENT:** the system's holdout delta decreases
monotonically with distance from the meta-training prior, and the boundary
where delta drops to zero is a structural property of the prior (not noise).

## Method

Meta-train on the standard task distribution. Evaluate on a gradient of
held-out user patterns, ordered by distance from the prior:

1. **Very similar:** new user, pattern nearly identical to a meta-training
   pattern (e.g., trained on `YYYY-MM-DD` formatting, test on `DD-MM-YYYY`).
2. **Same category:** new user, pattern from the same broad category but not
   seen in training (e.g., trained on date formatting rules, test on a number
   formatting rule).
3. **Different category, normal:** new user, pattern from a different category
   but still plausible user behavior (e.g., trained on formatting, test on a
   tone/style preference).
4. **Idiosyncratic:** new user, pattern that is unusual but deterministic
   (e.g., always reverses word order).
5. **Prior-contradicting:** new user, pattern that actively contradicts the
   meta-training prior (e.g., the prior favors standard formatting, but this
   user wants non-standard).

## Measure

Holdout delta at each level. Plot the generalization gradient.

## Success criterion

Not a single number. The result is the **boundary map**: which levels
transfer, which do not, and whether the boundary aligns with a structural
property.

## Kill criterion

- If the system fails at level 1, the meta-learner has not acquired a useful
  prior. It cannot even interpolate.
- If the system passes levels 1–2 but fails at 3, the prior is category-bound.
  This is a coverage problem (see R24.5), not a mechanism problem.
- If the system passes levels 1–4 but fails at 5, the prior is strong but
  rigid. This is a tunable problem (regularization, prior width).
- If the system passes all 5 levels, you have a genuinely strong result.

## Cost

Moderate. Requires generating a few user-pattern families, but each is small.
Can be done on the toy model first.

## Why this matters

This replaces the vague "unseen rules" language with a measurable
generalization profile. It also gives you an honest way to report partial
success: "the system generalizes within a family but not across families" is
a real, publishable, useful finding. The boundary map is more scientifically
informative than a single delta — it tells you *what the system learned*, not
just *whether it learned something*.

## Provenance

Proposed 2026-07-26 from external review. Not yet human-approved. The full
chat transcript is at `chat-export-1785143922754.json`.
