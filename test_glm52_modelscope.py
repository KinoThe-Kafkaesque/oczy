#!/usr/bin/env python3
"""Test inference for ModelScope-hosted GLM-5.2.

Run:
    export MODELSCOPE_TOKEN="<your-token>"
    python3 test_glm52_modelscope.py

Uses the OpenAI-compatible endpoint:
    https://api-inference.modelscope.ai/v1
and model ID:
    zai-org/GLM-5.2
"""
from __future__ import annotations

import os
import sys
import time

from openai import APIError, OpenAI

BASE_URL = "https://api-inference.modelscope.ai/v1"
MODEL_ID = "zai-org/GLM-5.2"


def get_client() -> OpenAI:
    token = os.environ.get("MODELSCOPE_TOKEN")
    if not token:
        print(
            "ERROR: MODELSCOPE_TOKEN environment variable is not set.",
            file=sys.stderr,
        )
        sys.exit(1)
    return OpenAI(base_url=BASE_URL, api_key=token)


def chat(
    client: OpenAI,
    messages: list[dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.7,
    stream: bool = True,
) -> dict[str, object]:
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
    )

    if not stream:
        choice = response.choices[0]
        return {
            "content": choice.message.content,
            "reasoning": getattr(choice.message, "reasoning_content", None),
            "usage": response.usage,
        }

    content = ""
    reasoning = ""
    finish_reason = None
    for chunk in response:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.delta.reasoning_content:
            reasoning += choice.delta.reasoning_content
        if choice.delta.content:
            content += choice.delta.content
        if choice.finish_reason:
            finish_reason = choice.finish_reason

    return {
        "content": content,
        "reasoning": reasoning,
        "usage": None,
        "finish_reason": finish_reason,
    }


def main() -> None:
    client = get_client()
    messages = [
        {
            "role": "user",
            "content": (
                "Hello, GLM-5.2! Please briefly introduce yourself "
                "and mention the size of your context window."
            ),
        }
    ]

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Attempt {attempt}/{max_attempts}: calling {MODEL_ID} ...")
            result = chat(client, messages, max_tokens=200, stream=True)
            break
        except APIError as exc:
            print(f"Attempt {attempt} failed: {type(exc).__name__}: {exc}")
            if attempt == max_attempts:
                raise
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError("All attempts failed")

    if result.get("reasoning"):
        print("\n--- reasoning ---")
        print(result["reasoning"])

    print("\n--- assistant content ---")
    print(result["content"])

    print("\n--- metadata ---")
    print(f"finish_reason: {result.get('finish_reason')}")
    if result.get("usage"):
        print(f"usage: {result['usage']}")
    else:
        print("usage: not returned in streaming mode")


if __name__ == "__main__":
    main()
