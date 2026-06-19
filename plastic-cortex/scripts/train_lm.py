"""Offline training script for the tiny NumPy LM PlasticCortex.

This script fits a CharTokenizer on a plain-text corpus and trains an
LMPlasticCortex with SGD, printing per-epoch average loss. After training it
saves the tokenizer and the model and runs a small smoke-test generation.

Adaptive mode:
    uv run python plastic-cortex/scripts/train_lm.py --adaptive

Adaptive mode first probes the existing checkpoint (if any) to generate a
curriculum targeted at the model's current uncertainty / novelty level, then
mixes that curriculum with the base corpus for training.

Auto-grow mode:
    uv run python plastic-cortex/scripts/train_lm.py --auto-grow

If early stopping fires and the loss is still above --grow-loss-threshold,
the model expands its hidden dimension by --grow-factor and resumes training
(up to five growth phases).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

# Make the trainer importable/runnable from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plastic_cortex.char_tokenizer import CharTokenizer
from plastic_cortex.lm_cortex import LMPlasticCortex


# Keep inline so adaptive mode can use the same templates without a subprocess.
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
    ],
}


def _resolve_default(path_str: str) -> Path:
    """Return a usable path whether running from the repo root or package root."""
    candidate = Path(path_str)
    if candidate.exists():
        return candidate
    package_root = Path(__file__).resolve().parent.parent
    parts = candidate.parts
    if parts and parts[0] == package_root.name:
        return package_root.joinpath(*parts[1:])
    return candidate


def load_corpus(path: Path) -> list[str]:
    """Load non-empty lines from the corpus text file."""
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line and not line.startswith("# ")]


def _score_item(model: LMPlasticCortex, prompt: str, target: str) -> dict[str, float]:
    """Return a difficulty/teaching-potential score for a curriculum item."""
    full_text = f"{prompt} {target}"
    uncertainty = float(model.uncertainty(full_text))
    novelty = float(model.novelty(full_text))
    prompt_uncertainty = float(model.uncertainty(prompt))
    zpd = (
        min(uncertainty, 4.0) / 4.0
        + novelty
        - 0.5 * max(0.0, 2.0 - uncertainty)
        - 0.3 * max(0.0, prompt_uncertainty - 3.0)
    )
    return {
        "uncertainty": uncertainty,
        "novelty": novelty,
        "prompt_uncertainty": prompt_uncertainty,
        "zpd": zpd,
    }


def generate_adaptive_lines(
    model: LMPlasticCortex,
    budget: int = 40,
    seed: int = 0,
) -> list[str]:
    """Build a list of full prompt-target sentences targeted at *model* level."""
    random.seed(seed)
    all_items: list[dict[str, Any]] = []
    for level_name, templates in CURRICULUM_TEMPLATES.items():
        for template in templates:
            prompt = template["prompt"]
            for target in template["targets"]:
                scores = _score_item(model, prompt, target)
                all_items.append(
                    {
                        "level": level_name,
                        "text": f"{prompt} {target}",
                        **scores,
                    }
                )

    all_items.sort(key=lambda x: x["zpd"], reverse=True)
    per_level: dict[str, list[dict[str, Any]]] = {}
    for item in all_items:
        per_level.setdefault(item["level"], []).append(item)

    min_per_level = max(1, budget // (2 * len(per_level)))
    selected: list[dict[str, Any]] = []
    for items in per_level.values():
        selected.extend(items[:min_per_level])

    selected_ids = {id(item) for item in selected}
    for item in all_items:
        if len(selected) >= budget:
            break
        if id(item) not in selected_ids and item["zpd"] > 0.0:
            selected.append(item)

    level_order = list(CURRICULUM_TEMPLATES.keys())
    selected.sort(key=lambda x: (level_order.index(x["level"]), -x["zpd"]))
    return [item["text"] for item in selected]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny NumPy LM on a character-level corpus."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("plastic-cortex/data/default_corpus.txt"),
        help="Path to the plain-text corpus (default: plastic-cortex/data/default_corpus.txt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="SGD learning rate (default: 0.01).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Recurrent hidden dimension (default: 128).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("plastic-cortex/checkpoints/lm"),
        help="Directory to write tokenizer.json and model.pkl (default: plastic-cortex/checkpoints/lm).",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Generate an adaptive curriculum from the existing checkpoint and mix it in.",
    )
    parser.add_argument(
        "--curriculum-budget",
        type=int,
        default=40,
        help="Number of adaptive curriculum items to generate (default: 40).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load an existing model.pkl from --outdir before training.",
    )
    parser.add_argument(
        "--auto-grow",
        action="store_true",
        help="If early stopping hits and loss is still high, increase hidden_dim and keep training.",
    )
    parser.add_argument(
        "--grow-factor",
        type=float,
        default=1.5,
        help="Multiplier for hidden_dim on auto-grow (default: 1.5).",
    )
    parser.add_argument(
        "--max-hidden-dim",
        type=int,
        default=1024,
        help="Ceiling for auto-grow hidden_dim (default: 1024).",
    )
    parser.add_argument(
        "--grow-loss-threshold",
        type=float,
        default=2.5,
        help="Loss above which auto-grow is considered (default: 2.5).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    corpus_path = _resolve_default(str(args.corpus))
    base_lines = load_corpus(corpus_path)
    if not base_lines:
        print(f"Corpus is empty: {corpus_path}", file=sys.stderr)
        return 1

    outdir = _resolve_default(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    tokenizer = CharTokenizer()

    all_lines = list(base_lines)

    # If adaptive/resume is requested, try to load a checkpoint.
    model: LMPlasticCortex | None = None
    model_path = outdir / "model.pkl"
    if args.resume or args.adaptive:
        if model_path.exists():
            model = LMPlasticCortex.load(model_path)
            print(f"Resumed checkpoint from {model_path}")

    adaptive_lines: list[str] = []
    if args.adaptive:
        if model is None:
            print(
                "[adaptive] no checkpoint found; generating curriculum from fresh model",
                file=sys.stderr,
            )
            tokenizer.fit(base_lines)
            model = LMPlasticCortex(
                {"hidden_dim": args.hidden_dim, "vocab_size": tokenizer.vocab_size, "seed": 42}
            )
        print("[adaptive] generating targeted curriculum...")
        adaptive_lines = generate_adaptive_lines(
            model, budget=args.curriculum_budget, seed=42
        )
        print(f"[adaptive] generated {len(adaptive_lines)} targeted lines")
        # Incorporate adaptive lines before final tokenizer fit so their tokens
        # are part of the model's vocabulary.
        chunk = max(1, len(base_lines) // max(1, len(adaptive_lines)))
        idx = 0
        for line in adaptive_lines:
            idx = min(idx, len(all_lines))
            all_lines.insert(idx, line)
            idx += chunk + 1

    # Fit tokenizer on the actual corpus (base + adaptive) and build the model
    # with the real vocabulary size.
    tokenizer.fit(all_lines)
    tokenizer.save(outdir / "tokenizer.json")

    # Fresh start if no checkpoint was loaded and no fresh model was made above.
    if model is None:
        config = {
            "hidden_dim": args.hidden_dim,
            "vocab_size": tokenizer.vocab_size,
            "seed": 42,
        }
        model = LMPlasticCortex(config)

    model.tokenizer = tokenizer

    print(f"Corpus: {corpus_path} ({len(base_lines)} base + {len(adaptive_lines)} adaptive lines, vocab_size={tokenizer.vocab_size})")

    global_best_loss = float("inf")
    global_best_path = outdir / "model_global_best.pkl"
    patience = 0
    grow_phase = 0
    max_grow_phases = 5
    min_improvement_ratio = 0.01  # at least 1% relative gain to justify more capacity
    last_phase_best = float("inf")

    while grow_phase <= max_grow_phases:
        phase_best_loss = float("inf")
        phase_best_path = outdir / f"model_phase_{grow_phase:02d}_best.pkl"
        patience = 0

        for epoch in range(1, args.epochs + 1):
            random.shuffle(all_lines)
            epoch_loss = 0.0
            for line in all_lines:
                model.reset_state()
                epoch_loss += model.train_step(line, lr=args.lr)
            avg_loss = epoch_loss / len(all_lines)
            global_epoch = epoch + grow_phase * args.epochs
            print(f"Epoch {global_epoch:03d}: hidden={model.hidden_dim} avg_loss={avg_loss:.6f}")

            if avg_loss < phase_best_loss:
                phase_best_loss = avg_loss
                patience = 0
                model.save(phase_best_path)
            else:
                patience += 1
                if patience >= 5:
                    print(f"Early stopping at epoch {global_epoch} (no improvement for 5 epochs).")
                    break

        # Update global best across all phases.
        if phase_best_path.exists() and phase_best_loss < global_best_loss:
            global_best_loss = phase_best_loss
            global_best_path.write_bytes(phase_best_path.read_bytes())

        can_grow = (
            args.auto_grow
            and grow_phase < max_grow_phases
            and int(model.hidden_dim * args.grow_factor) <= args.max_hidden_dim
        )
        # Only consider further growth if the previous expansion actually helped.
        if grow_phase > 0:
            improvement = (last_phase_best - phase_best_loss) / last_phase_best
            if improvement < min_improvement_ratio:
                print(f"\n[auto-grow] improvement {improvement:.2%} below {min_improvement_ratio:.0%}; stopping.")
                can_grow = False

        if can_grow:
            new_dim = min(int(model.hidden_dim * args.grow_factor), args.max_hidden_dim)
            print(f"\n[auto-grow phase {grow_phase + 1}] phase_best={phase_best_loss:.4f}; expanding hidden_dim: {model.hidden_dim} -> {new_dim}")
            model = model.grow(new_dim)
            last_phase_best = phase_best_loss
            grow_phase += 1
        else:
            break

    final_path = outdir / "model.pkl"
    if global_best_path.exists():
        final_path.write_bytes(global_best_path.read_bytes())
        print(f"Saved best model (loss={global_best_loss:.4f}) to {final_path}")
    else:
        model.save(final_path)
        print(f"Saved model to {final_path}")


    # Smoke test: generate a response for "hello".
    model.reset_state()
    prompt = "hello"
    response = model.answer(prompt, max_tokens=100, temperature=0.8)
    print(f"\nSmoke test -- prompt: {prompt!r}")
    print(f"Generated: {response!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
