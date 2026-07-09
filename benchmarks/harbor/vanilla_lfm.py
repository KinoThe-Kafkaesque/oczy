#!/usr/bin/env python3
"""Vanilla LFM agent wrapper — raw LFM2.5-1.2B without Oczy augmentation.

Uses the same instruct prompt format but WITHOUT knowledge retrieval.
"""

import os
import sys
from pathlib import Path

from llama_cpp import Llama


MODEL_FILE = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"


def _find_model() -> str:
    env = os.environ.get("OCZY_MODEL_PATH", "")
    if env and Path(env).exists():
        return env
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    for root, _dirs, files in os.walk(str(cache)):
        if MODEL_FILE in files:
            return os.path.join(root, MODEL_FILE)
    local = Path.home() / ".cache" / "oczy" / "models" / MODEL_FILE
    if local.exists():
        return str(local)
    raise FileNotFoundError(f"GGUF model not found. Set OCZY_MODEL_PATH.")


def main() -> int:
    question = sys.stdin.read().strip()
    if not question:
        print("ERROR: no question provided on stdin", file=sys.stderr)
        return 1

    model_path = _find_model()
    llm = Llama(model_path=model_path, n_ctx=2048, n_threads=8, verbose=False)

    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    output = llm(
        prompt,
        max_tokens=128,
        temperature=0.0,
        stop=["<|user|>", "\n\n"],
    )
    answer = output["choices"][0]["text"].strip()
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
