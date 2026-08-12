#!/usr/bin/env python3
"""Frontier-teacher validity check for R18's teacher gate (proposal §4a).

Cheap first check before any full loop: does the **OpenRouter frontier
teacher** (deepseek/deepseek-v4-flash-0731, provider pinned to DeepSeek)
clear R18's registered `teacher_dev_delta >= 0.2` admission criterion on the
same stage dev facts, in the same role the 0.5B-prefix teacher failed
(measured 0.1765 < 0.2)?

Teacher-delivery: the strategy:
    teacher = OpenRouterTeacher / probe_matches (same scorer as R18)

The gate formula is R18's registered formula, with `vanilla` selectable:

  * `--vanilla teacher` (default, self-contained, no local model):
    vanilla = teacher answering the probe with NO correction in context.
    delta = (teacher_with_correction - teacher_without_correction) / N.
    This is the pure "can the frontier teacher express the fact?" signal.

  * `--vanilla local` (exact-registered formula):
    vanilla = the frozen local organ (HFDriver) answering with no prefix;
    teacher = frontier teacher with correction in context.
    Delta is on the same scale as the recorded R18 gate (0.1765).

  * `--vanilla none`: report only absolute with-correction accuracy.

Reads only the eval/v2 dev split (no holdout access). This is a diagnostic,
not an experiment run: it changes nothing in eval/v2 and never edits the
pre-registered R18/R19 specs.

Usage:
    python scripts/teacher_validity_check.py [--stage stage_0_grounding] \
        [--vanilla teacher|local|none] [--limit N] [--seeds S]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Allow running from any cwd without the package installed.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.v2 import verify_manifest  # noqa: E402
from oczy.eval_v2.scoring import probe_matches  # noqa: E402
from oczy.experiments.organism_curriculum.dataset import (  # noqa: E402
    STAGE_ORDER,
    build_curriculum,
    split_probes,
)
from oczy.lm.openrouter_teacher import (  # noqa: E402
    OpenRouterTeacher,
    OpenRouterTeacherConfig,
)


def _run(args: argparse.Namespace) -> int:
    verify_manifest()

    stage = build_curriculum(stage_names=(args.stage,))[0]
    dev_ids, _holdout = split_probes(stage, fraction=0.3, salt="v2")

    teacher = OpenRouterTeacher(
        OpenRouterTeacherConfig(max_tokens=args.max_tokens, seed=args.seed)
    )
    print(f"# teacher: {teacher.config.describe()}", file=sys.stderr)

    # Optional local vanilla (only for exact-registered formula).
    driver = None
    if args.vanilla == "local":
        from oczy.lm.hf_driver import HFDriver

        driver = HFDriver.load()
        print("# local vanilla: loaded HFDriver", file=sys.stderr)

    dev_episode_ids = {pid.split("|")[0] for pid in dev_ids}
    dev_episodes = [ep for ep in stage.episodes if ep.id in dev_episode_ids]

    t_correct = 0
    t_total = 0
    v_correct = 0
    rows: list[dict[str, Any]] = []

    for ep in dev_episodes:
        for probe in ep.probes:
            pid = f"{ep.id}|{probe.request}|{probe.category}"
            if pid not in dev_ids:
                continue
            ans_t = teacher.answer_probe(
                probe.request, correction=ep.correction_utterance, max_tokens=args.max_tokens
            )
            correct_t = bool(ans_t.strip()) and probe_matches(ans_t, probe, ep)
            if correct_t:
                t_correct += 1
            t_total += 1

            if args.vanilla == "teacher":
                ans_v = teacher.answer_probe(
                    probe.request, correction=None, max_tokens=args.max_tokens
                )
                correct_v = probe_matches(ans_v, probe, ep)
            elif args.vanilla == "local":
                assert driver is not None
                ans_v = driver.generate(probe.request, max_tokens=args.max_tokens)
                correct_v = bool(ans_v.strip()) and probe_matches(ans_v, probe, ep)
            else:  # none
                ans_v, correct_v = None, False
            if correct_v:
                v_correct += 1

            rows.append({
                "episode": ep.id,
                "probe": probe.request,
                "expected": probe.expected,
                "match": probe.match_mode,
                "teacher_correct": correct_t,
                "vanilla_correct": correct_v,
                "teacher_answer": ans_t,
                "vanilla_answer": ans_v,
            })
            print(
                f"+ {ep.id} | {probe.request!r} | teacher={ans_t!r} "
                f"({'OK' if correct_t else 'miss'}) vanilla={'OK' if correct_v else 'miss'}"
            )
            if args.limit and t_total >= args.limit:
                break
        if args.limit and t_total >= args.limit:
            break

    teacher_delta = (t_correct - v_correct) / max(t_total, 1)
    teacher_abs = t_correct / max(t_total, 1)

    print("\n==== teacher validity gate ====")
    print(f"teacher_model          = {teacher.config.model}")
    print(f"provider_pinning       = {list(teacher.config.provider_only)}")
    print(f"vanilla_source         = {args.vanilla}")
    print(f"dev_probes            = {t_total}")
    print(f"teacher_correct       = {t_correct}/{t_total} = {teacher_abs:.4f}")
    print(f"vanilla_correct       = {v_correct}/{t_total} = {v_correct/max(t_total,1):.4f}")
    print(f"teacher_dev_delta     = {teacher_delta:.4f}  (gate: >= 0.2000)")
    print("R18_recorded_delta    = 0.1765 (0.5B-prefix teacher, all seeds)")
    print(f"GATE_CLEARED          = {teacher_delta >= 0.2}")
    print(f"usage_tokens          = {sum((r.get('usage') or {}).get('total_tokens', 0) for r in teacher.log)}")
    print(f"ASI teacher_dev_delta={teacher_delta:.6f}")
    print(f"ASI teacher_abs={teacher_abs:.6f}")
    print(f"ASI teacher_model={teacher.config.model}")
    print(f"ASI teacher_provider_only={','.join(teacher.config.provider_only)}")
    print(f"ASI teacher_vanilla_source={args.vanilla}")
    print(f"ASI teacher_dev_probes={t_total}")
    print(f"ASI stage={args.stage}")

    if driver is not None:
        driver.close()
    teacher.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", type=str, default="stage_0_grounding",
                        choices=STAGE_ORDER, help="curriculum stage for dev facts")
    parser.add_argument("--vanilla", choices=("teacher", "local", "none"), default="teacher",
                        help="vanilla baseline source for teacher_dev_delta")
    parser.add_argument("--limit", type=int, default=0, help="max dev probes (0 = all)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="teacher answer max tokens (default: UNSET = "
                        "unbound / provider default; do not cap at 32, it "
                        "truncates answers)")
    parser.add_argument("--seed", type=int, default=0,
                        help="teacher sampling seed (best-effort reproducibility)")
    args = parser.parse_args(argv)
    args = parser.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
