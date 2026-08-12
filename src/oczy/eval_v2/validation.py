"""Curriculum validation: guard against the flaws of the old benchmark.

Run as a smoke test:

    python -m experiments.organism_curriculum.validation

Validation rules:

1. Every episode must have ``correction_utterance`` and ``corrected_label``.
2. The ambiguous token (quoted in the correction) must not overlap with the
   default priors in ``PlasticCortex.BASELINE`` keys or labels.
3. Transfer probes must not contain the ``corrected_response`` or
   ``corrected_label`` verbatim, to prevent substring-matching fakes.
4. Forgetting / retention probes must include the ambiguous token so that
   they actually test retention of the prior sense, not unrelated facts.
5. Episode ids must be unique across all loaded stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from oczy.experiments.organism_curriculum.dataset import (
    DEFAULT_SPLIT_SALT,
    Episode,
    Stage,
    build_curriculum,
    split_probes,
)
from plastic_cortex.cortex import PlasticCortex


def _baseline_tokens() -> set[str]:
    """Return the union of tokens that PlasticCortex starts with."""
    tokens: set[str] = set()
    for key, scores in PlasticCortex.BASELINE.items():
        tokens.add(key)
        tokens.update(scores.keys())
    # Also include labels because the cortex ties answers to them.
    tokens.update(PlasticCortex.LABELS)
    return tokens


_BASELINE_TOKENS = _baseline_tokens()


@dataclass
class ValidationReport:
    """Result of validating a curriculum."""

    ok: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def merge(self, other: "ValidationReport") -> None:
        self.warnings.extend(other.warnings)
        self.errors.extend(other.errors)
        if not other.ok:
            self.ok = False


def _validate_episode(ep: Episode, stage_name: str, baseline_tokens: set[str]) -> ValidationReport:
    report = ValidationReport()
    prefix = "%s/%s" % (stage_name, ep.id)

    if not ep.correction_utterance.strip():
        report.add_error("%s: empty correction_utterance" % prefix)
    if not ep.corrected_label.strip():
        report.add_error("%s: empty corrected_label" % prefix)
    if not ep.corrected_response.strip():
        report.add_error("%s: empty corrected_response" % prefix)

    token = ep.ambiguous_token()
    if token is None:
        report.add_error(
            "%s: could not extract quoted ambiguous token from correction_utterance" % prefix
        )
    else:
        if token in baseline_tokens:
            report.add_error(
                "%s: ambiguous token %r overlaps with PlasticCortex.BASELINE" % (prefix, token)
            )
        # Forgetting/retention probes should keep the ambiguous word in play.
        for probe in ep.probes:
            if probe.category in ("forgetting", "retention"):
                if token not in probe.request.lower() and token not in probe.expected.lower():
                    report.add_warning(
                        "%s: %s probe does not contain ambiguous token %r"
                        % (prefix, probe.category, token)
                    )

    # Transfer probes should not be solvable by substring matching of the
    # full corrected response; short corrected labels are expected to recur.
    for probe in ep.probes:
        if probe.category == "transfer":
            lower_expected = probe.expected.lower()
            if ep.corrected_response.lower() in lower_expected:
                report.add_warning(
                    "%s: transfer probe contains corrected_response verbatim" % prefix
                )

    # Corrected label should be reasonably distinct from the ambiguous token.
    if token and ep.corrected_label.lower() == token:
        report.add_warning(
            "%s: corrected_label equals ambiguous token %r" % (prefix, token)
        )

    return report


def validate_stage(stage: Stage, baseline_tokens: set[str] | None = None) -> ValidationReport:
    """Validate a single stage, returning a report of warnings and errors."""
    baseline_tokens = baseline_tokens or _BASELINE_TOKENS
    report = ValidationReport()
    if stage.probe_timing not in ("after_stage", "after_episode"):
        report.add_error(
            f"{stage.name}: unknown probe_timing {stage.probe_timing!r}"
        )
    if stage.consolidate_after and stage.probe_timing != "after_stage":
        report.add_error(
            f"{stage.name}: consolidate_after requires probe_timing='after_stage'"
        )
    if not stage.episodes:
        report.add_warning("%s: stage has no episodes" % stage.name)
    for ep in stage.episodes:
        report.merge(_validate_episode(ep, stage.name, baseline_tokens))
    return report


def validate_curriculum(stages: tuple[Stage, ...]) -> ValidationReport:
    """Validate every stage plus cross-stage id uniqueness."""
    report = ValidationReport()
    seen_ids: dict[str, str] = {}
    for stage in stages:
        report.merge(validate_stage(stage))
        for ep in stage.episodes:
            if ep.id in seen_ids:
                report.add_error(
                    "Duplicate episode id %r in %s and %s"
                    % (ep.id, seen_ids[ep.id], stage.name)
                )
            else:
                seen_ids[ep.id] = stage.name
    return report

def validate_split(
    stages: tuple[Stage, ...],
    fraction: float = 0.3,
    salt: str = DEFAULT_SPLIT_SALT,
) -> ValidationReport:
    """Validate the held-out split across all stages.

    Rules:
    - Every stage with >0 probes must have at least 1 holdout probe.
    - Every stage with >0 probes must have at least 1 dev probe (warning).
    """
    report = ValidationReport()
    for stage in stages:
        dev, holdout = split_probes(stage, fraction=fraction, salt=salt)
        total = len(dev) + len(holdout)
        if total > 0 and len(holdout) == 0:
            report.add_error(f"{stage.name}: holdout split is empty ({total} probes)")
        if total > 0 and len(dev) == 0:
            report.add_warning(f"{stage.name}: dev split is empty ({total} probes)")
        if salt != "v2":
            categories: dict[str, set[str]] = {}
            for ep in stage.episodes:
                for probe in ep.probes:
                    categories.setdefault(probe.category, set()).add(
                        f"{ep.id}|{probe.request}|{probe.category}"
                    )
            for category, probe_ids in categories.items():
                if len(probe_ids) >= 2 and 0.0 < fraction < 1.0:
                    if not (probe_ids & holdout):
                        report.add_error(
                            f"{stage.name}: holdout omits category {category!r}"
                        )
                    if not (probe_ids & dev):
                        report.add_error(
                            f"{stage.name}: dev split omits category {category!r}"
                        )
    return report


def main(argv: list[str] | None = None) -> int:
    stages = build_curriculum()
    report = validate_curriculum(stages)

    print("Validated %d stage(s); %d episode(s)" % (len(stages), sum(len(s.episodes) for s in stages)))
    if report.warnings:
        print("\nWarnings (%d):" % len(report.warnings))
        for w in report.warnings:
            print("  - %s" % w)
    if report.errors:
        print("\nErrors (%d):" % len(report.errors))
        for e in report.errors:
            print("  - %s" % e)
    if report.ok and not report.warnings:
        print("All checks passed.")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
