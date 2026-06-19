"""Generate an adaptive curriculum targeted at the current LMPlasticCortex level.

The generator probes a model checkpoint with a bank of template sentences at
multiple difficulty levels, ranks each item by a "zone-of-proximal-development"
score (moderate uncertainty + moderate novelty), and writes a curriculum text
file ready for ``train_lm.py``.

Usage:
    uv run python plastic-cortex/scripts/generate_curriculum.py \
        --checkpoint plastic-cortex/checkpoints/lm/model.pkl \
        --output plastic-cortex/data/adaptive_curriculum.txt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

# Make script runnable from repo root or package root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plastic_cortex.lm_cortex import LMPlasticCortex


# Difficulty-level template banks.  The model should learn to complete/repeat
# these forms.  Each item is a prompt + expected continuation pair.
CURRICULUM_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "level_0_greetings": [
        {"prompt": "hello", "targets": ["Hello!", "Hi!"]},
        {"prompt": "how are you", "targets": ["I am well, thank you.", "Doing fine."]},
        {"prompt": "good morning", "targets": ["Good morning!", "Morning!"]},
        {"prompt": "goodbye", "targets": ["Goodbye!", "See you later."]},
        {"prompt": "thank you", "targets": ["You are welcome.", "Glad to help."]},
    ],
    "level_1_identity": [
        {
            "prompt": "Who are you?",
            "targets": [
                "I am a small language model built from NumPy.",
                "I am an assistant that learns from corrections.",
            ],
        },
        {
            "prompt": "What is your purpose?",
            "targets": [
                "My purpose is to assist and learn from corrections.",
                "I exist to help and update my understanding.",
            ],
        },
        {
            "prompt": "What is the project name?",
            "targets": ["The project is called Oczy.", "This project is Oczy."],
        },
    ],
    "level_2_concepts": [
        {
            "prompt": "What is a plastic cortex?",
            "targets": ["A plastic cortex is a recurrent module that stores fast conversation weights."],
        },
        {
            "prompt": "What is a fast weight?",
            "targets": ["A fast weight is a short-term memory entry that boosts specific token logits."],
        },
        {
            "prompt": "How does correction work?",
            "targets": ["You give me a trigger phrase and an expected answer, and my fast weights boost the expected tokens."],
        },
        {
            "prompt": "What is cross-entropy loss?",
            "targets": ["Cross-entropy loss measures how surprised the model is by the true next token."],
        },
    ],
    "level_3_disambiguation": [
        {
            "prompt": 'When I say "branch", I mean a',
            "targets": ["git branch, not a tree branch."],
        },
        {
            "prompt": '"Batch" here means a',
            "targets": ["group of training examples, not a baked good."],
        },
        {
            "prompt": '"Model" means a',
            "targets": ["machine-learning model, not a fashion model."],
        },
        {
            "prompt": '"Profile" means a',
            "targets": ["resource profile, not a social-media profile."],
        },
    ],
    "level_4_open": [
        {
            "prompt": "Explain the Oczy organism in one sentence.",
            "targets": [
                "Oczy is a modular plastic world-model agent that metabolizes experience into weights instead of memorizing raw traces."
            ],
        },
        {
            "prompt": "Why do we use NumPy only?",
            "targets": ["NumPy keeps the dependency tree small and makes every gradient transparent."],
        },
        {
            "prompt": "How do I know if the model learned?",
            "targets": ["Compare pre-test and post-test scores on a hold-out subset."],
        },
        {
            "prompt": "What should I do if the model confuses two meanings?",
            "targets": [
                'Provide a correction that names the ambiguous word and gives the correct context, like "batch means ML batch".'
            ],
        },
    ],
}


def _score_item(
    model: LMPlasticCortex,
    full_text: str,
    target: str,
    uncertainty_weight: float = 1.0,
    novelty_weight: float = 1.0,
) -> dict[str, float]:
    """Return a difficulty/teaching-potential score for a curriculum item."""
    uncertainty = float(model.uncertainty(full_text))
    novelty = float(model.novelty(full_text))
    # How uncertain is the model when it only sees the prompt?  Very high
    # prompt uncertainty means the target is currently unreachable.
    prompt_text = full_text[: len(full_text) - len(target)] if len(target) <= len(full_text) else full_text
    prompt_uncertainty = float(model.uncertainty(prompt_text))

    # ZPD score: moderate uncertainty and moderate novelty are best.
    zpd = (
        uncertainty_weight * min(uncertainty, 4.0) / 4.0
        + novelty_weight * novelty
        - 0.5 * max(0.0, 2.0 - uncertainty)  # penalty for being too easy
        - 0.3 * max(0.0, prompt_uncertainty - 3.0)  # penalty for unreachable target
    )
    return {
        "uncertainty": uncertainty,
        "novelty": novelty,
        "prompt_uncertainty": prompt_uncertainty,
        "zpd": zpd,
    }


def _build_items(
    model: LMPlasticCortex, level_name: str, templates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Generate scored items from one difficulty level."""
    items: list[dict[str, Any]] = []
    for template in templates:
        prompt = template["prompt"]
        for target in template["targets"]:
            full_text = f"{prompt} {target}"
            scores = _score_item(model, full_text, target)
            items.append(
                {
                    "level": level_name,
                    "prompt": prompt,
                    "target": target,
                    "text": full_text,
                    **scores,
                }
            )
    return items


def _select_items(
    all_items: list[dict[str, Any]], budget: int
) -> list[dict[str, Any]]:
    """Pick a diverse subset maximizing ZPD while keeping levels balanced."""
    # First, sort all items by ZPD descending.
    all_items = sorted(all_items, key=lambda x: x["zpd"], reverse=True)

    # Then enforce a per-level minimum so each level is represented.
    per_level: dict[str, list[dict[str, Any]]] = {}
    for item in all_items:
        per_level.setdefault(item["level"], []).append(item)

    selected: list[dict[str, Any]] = []
    min_per_level = max(1, budget // (2 * len(per_level)))
    remaining_budget = budget

    for level, items in per_level.items():
        take = min(min_per_level, len(items))
        selected.extend(items[:take])
        remaining_budget -= take

    # Fill remaining budget from the global ZPD ranking, avoiding duplicates.
    selected_ids = {id(item) for item in selected}
    for item in all_items:
        if remaining_budget <= 0:
            break
        if id(item) not in selected_ids and item["zpd"] > 0.0:
            selected.append(item)
            remaining_budget -= 1

    # Final curriculum: order by level, then by ZPD descending.
    level_order = list(CURRICULUM_TEMPLATES.keys())
    selected.sort(key=lambda x: (level_order.index(x["level"]), -x["zpd"]))
    return selected


def _write_curriculum(items: list[dict[str, Any]], path: Path) -> None:
    """Write the curriculum as a plain text file with metadata comments."""
    lines: list[str] = [
        "# Adaptive curriculum generated for LMPlasticCortex",
        "# Lines beginning with '# level' mark a new difficulty level.",
        "# Each non-comment line is a complete prompt-target sentence.",
        "",
    ]
    current_level: str | None = None
    for item in items:
        if item["level"] != current_level:
            current_level = item["level"]
            lines.append(f"# level={current_level} zpd={item['zpd']:.3f}")
        lines.append(item["text"])
        lines.append(
            f"# scores: uncertainty={item['uncertainty']:.2f} "
            f"novelty={item['novelty']:.2f} zpd={item['zpd']:.3f}"
        )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an adaptive curriculum for LMPlasticCortex."
    )
    parser.add_argument(
        "--checkpoint",
        default="plastic-cortex/checkpoints/lm/model.pkl",
        help="Path to a saved LMPlasticCortex checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="plastic-cortex/data/adaptive_curriculum.txt",
        help="Where to write the generated curriculum.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=40,
        help="Maximum number of curriculum items to keep (default: 40).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        print("Train a model first with:", file=sys.stderr)
        print("  uv run python plastic-cortex/scripts/train_lm.py", file=sys.stderr)
        return 1

    model = LMPlasticCortex.load(checkpoint_path)

    all_items: list[dict[str, Any]] = []
    for level_name, templates in CURRICULUM_TEMPLATES.items():
        all_items.extend(_build_items(model, level_name, templates))

    selected = _select_items(all_items, args.budget)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_curriculum(selected, output_path)

    print(f"Generated adaptive curriculum: {output_path}")
    print(f"Items selected: {len(selected)} / {len(all_items)}")
    print(f"Average ZPD score: {sum(i['zpd'] for i in selected)/max(1,len(selected)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
