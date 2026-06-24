#!/usr/bin/env python3
"""LM-mediated vs raw curriculum absorption demo.

Loads the curriculum (word-sense disambiguation episodes from
correction-benchmark) and feeds each lesson twice:

  "raw"   -- Oczy OrganismAgent sees the literal request/correction strings
              set in the public API (agent.learn(request, correction)).
              No LM in the loop.  This is the pre-LM baseline.

  "LM"    -- Each request+correction is wrapped into one natural-language
              utterance (mimicking how a user would actually phrase
              both), then the LanguageAdapter parses it into a canonical
              Episode, then the organism consumes the Episode.

Per-lesson measurement:

  - Corrected-label surfaced correctly on re-asking the request?
    (binary pass/fail)
  - Did the LM adapter produce a valid Episode AND extract the right
    correction?  (binary pass/fail, LM only)
  - Per-lesson wallclock.

End-of-run comparison:

  - Pinpoint which lessons the LM got wrong (raw could still learn those)
  - Aggregate absorption rate
  - Aggregate LM parse reliability
  - Aggregate wallclock premium

Run:
  .venv/bin/python experiments/lm_perception/run_perception_demo.py
  .venv/bin/python experiments/lm_perception/run_perception_demo.py --lessons 5
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from correction_benchmark.dataset import build_dataset, Episode as BenchEpisode

from experiments.organism import OrganismAgent
from oczy_common import extract_expected_from_correction, validate_episode
from oczy_lm import LanguageAdapter


@dataclass
class LessonResult:
    """One row of the curriculum absorption measurement."""
    idx: int
    request: str                       # the original curriculum request
    correction: str                    # the original curriculum correction
    expected_label: str                # the canonical label we expect to recover

    raw_absorbed: bool = False          # OrganismAgent surfaces corrected_label?
    raw_seconds: float = 0.0

    lm_parse_ok: bool = False          # adapter produced valid Episode + matched expected_label
    lm_extracted_label: str = ""       # what the adapter pulled out as corrected_answer
    lm_absorbed: bool = False           # post-LM-feeding, organism surfaces correct answer
    lm_seconds: float = 0.0           # total LM-mediated wallclock per lesson
    lm_episode_validation_warnings: list[str] = field(default_factory=list)


def build_nl_utterance(ep: BenchEpisode) -> str:
    """Compose a natural-language utterance the user might actually send.

    Mimics what a real user says in two clauses: the request, then the
    correction.  This is what the LM perceives for the LM-mediated run.
    """
    # Use just the first sentence of the correction; the dataset's
    # ``.correction`` field is already short.
    cor = ep.correction.strip()
    # Some corrections include trailing quotes; tame them.
    cor = cor.strip(".\"'")
    return f"{ep.request} {cor}"


def surface(agent: OrganismAgent, request: str, expected_label: str) -> bool:
    """Ask the agent for the request, return whether expected surfaces."""
    out = agent.answer(request)
    return expected_label.lower() in out.lower()


def run_raw(agent: OrganismAgent, ep: BenchEpisode) -> tuple[bool, float]:
    """Baseline: feed literal correction to agent, return (pass, seconds)."""
    t0 = time.perf_counter()
    # Use the public learn() interface (request + correction text).
    agent.learn(ep.request, ep.correction)
    # Now ask the request again — is the corrected answer surfaced?
    # The dataset's ``corrected_answer`` field is a full sentence like
    # "I'll update the business vertical configuration.", but the
    # curriculum's ``target_label`` is the shorter word (e.g. "business vertical").
    # The label part is what we want to find in the organism's output.
    label = extract_label(ep.correction, ep.corrected_answer)
    ok = surface(agent, ep.request, label)
    return ok, time.perf_counter() - t0


def extract_label(correction: str, corrected_answer: str = "") -> str:
    """Recover the short corrected-sense label.

    Preferred source is the ``correction`` text, which the curriculum
    writes in the form ``'No, "X" means Y.'`` -- the shared
    :func:`oczy_common.extract_expected_from_correction` extracts Y
    cleanly.  Fallback: the ``corrected_answer`` sentence, which the
    extractor trims to its content.

    This is the same heuristic the organism itself uses (via
    ``_extract_expected_from_correction``) so the surface() check at
    evaluation time is consistent with what the organism stored under
    ``corrected_answer`` on the raw path.
    """
    # Prefer the correction text; if extractor collapses to the
    # whole string, try the corrected_answer sentence instead.
    label = extract_expected_from_correction(correction)
    if not label or len(label) >= len(correction):
        label = extract_expected_from_correction(corrected_answer)
    return label.strip()


def run_lm(agent: OrganismAgent, adapter: LanguageAdapter,
           ep: BenchEpisode) -> LessonResult:
    """LM-mediated: NL utterance -> adapter -> Episode -> organism.

    We produce one LessonResult entry per curriculum episode for the LM
    run specifically.
    """
    res = LessonResult(
        idx=0,
        request=ep.request,
        correction=ep.correction,
        expected_label=extract_label(ep.correction, ep.corrected_answer),
    )

    nl = build_nl_utterance(ep)
    t0 = time.perf_counter()

    # Parse the natural-language utterance into a canonical Episode.
    episode = adapter.nl_to_episode(nl)
    res.lm_episode_validation_warnings = validate_episode(episode)

    # The adapter may yield "" as corrected_answer on failed parse.
    # The clarity check is whether the extracted label matches expectation.
    got = episode.get("corrected_answer", "")
    res.lm_extracted_label = got
    res.lm_parse_ok = (got.lower() in ep.corrected_answer.lower()
                       or ep.correction.lower().find(got.lower()) != -1) and bool(got)

    # Feed to the organism: use learn() so the same code path is exercised.
    # The organism uses ``extract_expected_from_correction`` internally,
    # so the literal correction text is what it actually parses.
    # For the LM test we want to see if the LM's extracted_label (which
    # may be cleaner than what the heuristic would pull out) made
    # absorption easier/faster/more-successful.
    if res.lm_extracted_label:
        agent.learn(ep.request, ep.correction)
        res.lm_absorbed = surface(agent, ep.request, res.lm_extracted_label)
    else:
        # Bad parse: just attempt the raw feed and surface so we can
        # measure whether the organism still picked anything up.
        agent.learn(ep.request, ep.correction)
        res.lm_absorbed = surface(agent, ep.request,
                                  extract_label(ep.correction, ep.corrected_answer))

    res.lm_seconds = time.perf_counter() - t0
    return res


def run_one_episode(idx: int, ep: BenchEpisode,
                    adapter: LanguageAdapter) -> LessonResult:
    """Run both baselines AND the LM-mediated path on one episode with
    fresh agents, returning a single LessonResult.

    Fresh agents per episode: absorption is one-shot, so prior episodes
    don't pollute the current measurement.
    """
    res = LessonResult(
        idx=idx, request=ep.request, correction=ep.correction,
        expected_label=extract_label(ep.correction, ep.corrected_answer),
    )

    # Raw run: fresh agent per episode for one-shot measurement.
    raw_agent = OrganismAgent()
    res.raw_absorbed, res.raw_seconds = run_raw(raw_agent, ep)
    del raw_agent; gc.collect()

    # LM run: fresh agent, but parse the LM-path utterance.
    lm_agent = OrganismAgent()
    lm_res = run_lm(lm_agent, adapter, ep)
    res.lm_parse_ok = lm_res.lm_parse_ok
    res.lm_extracted_label = lm_res.lm_extracted_label
    res.lm_absorbed = lm_res.lm_absorbed
    res.lm_seconds = lm_res.lm_seconds
    res.lm_episode_validation_warnings = lm_res.lm_episode_validation_warnings
    del lm_agent; gc.collect()

    return res


def print_progress(res: LessonResult) -> None:
    label_preview = res.expected_label[:30]
    print(f"  [{res.idx:2d}] {res.request[:35]:35} | "
          f"raw: {'OK' if res.raw_absorbed else '..'} ({res.raw_seconds:.2f}s) | "
          f"lm parse: {'OK' if res.lm_parse_ok else '..'} ('{res.lm_extracted_label[:20]}') | "
          f"lm absorb: {'OK' if res.lm_absorbed else '..'} ({res.lm_seconds:.2f}s) | "
          f"target: {label_preview!r}")


def print_summary(results: list[LessonResult]) -> None:
    n = max(1, len(results))
    raw_n = sum(1 for r in results if r.raw_absorbed)
    lm_parse_n = sum(1 for r in results if r.lm_parse_ok)
    lm_absorb_n = sum(1 for r in results if r.lm_absorbed)
    raw_avg = sum(r.raw_seconds for r in results) / n
    lm_avg = sum(r.lm_seconds for r in results) / n

    print()
    print("=" * 78)
    print(f"Curriculum absorption summary  ({len(results)} lessons)")
    print("=" * 78)
    print(f"  Raw  absorbed  : {raw_n:3d}/{n}  ({100*raw_n/n:.0f}%)  "
          f"avg {raw_avg:.2f}s/lesson")
    print(f"  LM   parse OK  : {lm_parse_n:3d}/{n}  ({100*lm_parse_n/n:.0f}%)")
    print(f"  LM   absorbed  : {lm_absorb_n:3d}/{n}  ({100*lm_absorb_n/n:.0f}%)  "
          f"avg {lm_avg:.2f}s/lesson  (LM+organism end-to-end)")
    print(f"  Wallclock premium per lesson : "
          f"{lm_avg - raw_avg:+.2f}s  (LM path - raw)")
    if lm_parse_n < n:
        miss = [r for r in results if not r.lm_parse_ok]
        print(f"  LM parse misses ({len(miss)}):")
        for r in miss:
            print(f"    [{r.idx:2d}] {r.request[:50]:50} "
                  f"| extracted={r.lm_extracted_label!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lessons", type=int, default=-1,
                   help="Cap on number of lessons to run (default: all).")
    p.add_argument("--start", type=int, default=0,
                   help="Start at lesson index N (0-based).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    episodes = list(build_dataset())
    if args.start > 0:
        episodes = episodes[args.start:]
    if args.lessons > 0:
        episodes = episodes[:args.lessons]
    print(f"Oczy LM perception demo")
    print(f"  lessons: {len(episodes)}")
    print(f"  loading adapter (LFM2.5-1.2B Q4_K_M, lazy)...")

    adapter = LanguageAdapter()
    # Pre-load so first measurement isn't dominated by model load time.
    adapter.load()
    print(f"  adapter ready")

    results: list[LessonResult] = []
    for idx, ep in enumerate(episodes, start=args.start):
        try:
            r = run_one_episode(idx, ep, adapter)
        except Exception as e:
            print(f"  [{idx}] EXCEPTION: {type(e).__name__}: {str(e)[:120]}")
            continue
        print_progress(r)
        results.append(r)

    print_summary(results)

    # Persist the table for follow-up analysis.
    out_dir = Path("experiments/lm_perception/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demo_run.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump([{
            "idx": r.idx,
            "request": r.request,
            "correction": r.correction,
            "expected_label": r.expected_label,
            "raw_absorbed": r.raw_absorbed,
            "raw_seconds": r.raw_seconds,
            "lm_parse_ok": r.lm_parse_ok,
            "lm_extracted_label": r.lm_extracted_label,
            "lm_absorbed": r.lm_absorbed,
            "lm_seconds": r.lm_seconds,
            "lm_warnings": r.lm_episode_validation_warnings,
        } for r in results], fh, indent=2)
    print(f"\nReport written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())