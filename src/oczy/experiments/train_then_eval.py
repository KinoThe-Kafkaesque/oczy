"""Train an OrganismAgent by walking it through the curriculum, then test generalization.

Usage:
    uv run python experiments/train_then_eval.py
    uv run python experiments/train_then_eval.py --epochs 3 --session work
    uv run python experiments/train_then_eval.py --seed 1 --eval-seed 2 --consolidate

The agent is presented with every acquisition episode as a learning opportunity:
answer, correction, answer again. After training it is evaluated on the full probe
battery (transfer, scope, forgetting). Finally the trained state is saved so it
can be loaded by chat.py with the same --session name.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from oczy.experiments.curriculum import build_curriculum
from oczy.experiments.eval_suite import EvalSuite
from oczy.experiments.organism import OrganismAgent


def _session_dir() -> Path:
    """Return a system-appropriate directory for Oczy session files."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "oczy" / "sessions"


def _session_path(name: str) -> Path:
    return _session_dir() / f"{name}.pkl"


def _normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _matches(answer: str, expected: str) -> bool:
    return _normalize(answer) == _normalize(expected)


def _train_agent(agent: OrganismAgent, curriculum: Any, epochs: int = 1) -> dict[str, Any]:
    """Present every acquisition episode to the agent as a learning sequence."""
    trace: list[dict[str, Any]] = []
    for epoch in range(epochs):
        for level in curriculum.levels():
            for episode in level.group.acquisition_episodes:
                before = agent.answer(episode.request)
                agent.correct(episode.correction, episode.corrected_answer)
                after = agent.answer(episode.request)
                trace.append(
                    {
                        "epoch": epoch + 1,
                        "level": level.name,
                        "request": episode.request,
                        "before": before,
                        "after": after,
                        "expected": episode.corrected_answer,
                        "fixed": _matches(after, episode.corrected_answer),
                    }
                )
    return {"episodes": trace, "total": len(trace)}


def _snapshot_scores(snapshot: Any) -> dict[str, float]:
    return {
        "transfer": snapshot.accuracy("transfer"),
        "scope": snapshot.accuracy("scope"),
        "forgetting": snapshot.accuracy("forgetting"),
        "identity": snapshot.accuracy("identity"),
    }


def _print_scores(scores: dict[str, float]) -> None:
    for key in ("transfer", "scope", "forgetting", "identity"):
        print(f"  {key.title():>10}: {scores[key]:.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train OrganismAgent on the curriculum and test generalization."
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used for training curriculum.")
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Seed for fresh eval curriculum (defaults to --seed).",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of passes over the curriculum."
    )
    parser.add_argument(
        "--session", default="trained", help="Session name to save the trained agent."
    )
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Consolidate raw traces before final eval.",
    )
    args = parser.parse_args(argv)

    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed

    train_curriculum = build_curriculum(seed=args.seed)
    eval_curriculum = build_curriculum(seed=eval_seed)

    agent = OrganismAgent()

    print(f"Train curriculum seed: {args.seed}; eval curriculum seed: {eval_seed}")
    print()

    print("Pre-training evaluation on eval curriculum (no learning):")
    eval_suite = EvalSuite(eval_curriculum, sense_match=True)
    pre_scores = _snapshot_scores(eval_suite.pre_test(agent))
    _print_scores(pre_scores)
    print()

    print(f"Training on {len(train_curriculum.levels())} levels for {args.epochs} epoch(s)...")
    train_trace = _train_agent(agent, train_curriculum, epochs=args.epochs)
    fixed_count = sum(ep["fixed"] for ep in train_trace["episodes"])
    print(f"Episodes presented: {train_trace['total']}; fixed after correction: {fixed_count}")
    print()

    if args.consolidate:
        print("Consolidating raw traces...")
        agent.consolidate()
        print()

    print("Post-training evaluation on eval curriculum:")
    post_scores = _snapshot_scores(eval_suite.post_test(agent))
    _print_scores(post_scores)
    print()

    print("Pre → Post deltas:")
    for key in ("transfer", "scope", "forgetting", "identity"):
        delta = post_scores[key] - pre_scores[key]
        direction = "+" if delta >= 0 else ""
        print(f"  {key.title():>10}: {pre_scores[key]:.4f} → {post_scores[key]:.4f} ({direction}{delta:+.4f})")
    print()

    session_path = _session_path(args.session)
    agent.save(session_path)
    print(f"Trained agent saved to: {session_path}")
    print(f"Chat with it: uv run python chat.py --session {args.session}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
