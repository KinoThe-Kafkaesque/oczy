"""OpenRouter frontier-teacher client (R18/R19 de-block, project decision 2026-07-26).

Routing decision (human, 2026-07-26):
    provider  = OpenRouter
    model     = ``deepseek/deepseek-v4-flash-0731``
    routing   = pinned to DeepSeek with NO cross-provider fallback
                (``provider.only=[...]``), so teacher outputs stay
                reproducible across runs like the version-pinned model
                manifests under ``infrastructure/kaggle/model_manifests/``.

API key resolution (never committed, never printed):
    1. ``OPENROUTER_API_KEY`` environment variable; else
    2. the Prime Agent's stored OpenRouter credential at
       ``~/.prime/agent/auth.json`` under key ``openrouter.key``.

Output length is **unbound by default** (``max_tokens=None`` -> the field is
omitted and the provider's max output applies); an artificial 32-token cap
truncates answers and would corrupt later co-learning relabeling.

The endpoint is OpenAI-compatible (``/chat/completions``), the same shape
that ``benchmarks/pi/proxy_server.py`` already speaks, so the teacher
adapter is a thin, testable client.

This is *wiring only*: it does not run any experiment, does not modify
eval/v2, thresholds, or gates.  The pre-registered R18/R19 specs remain
unchanged; use of this teacher is gated on the registered
``teacher_dev_delta >= 0.2`` admission criterion (see
``notes/2026-07-26_r18-r19_deblock_proposal.md``).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_PROVIDER_ONLY = ("DeepSeek",)
DEFAULT_AUTH_PATH = Path.home() / ".prime" / "agent" / "auth.json"

_RETRY_STATUSES = (429, 500, 502, 503, 504)


def resolve_openrouter_api_key(
    *,
    env: dict[str, str] | None = None,
    auth_path: Path | None = None,
) -> str:
    """Return the OpenRouter API key from the environment or Prime Agent auth.

    Resolution order:
      1. ``OPENROUTER_API_KEY`` env var
      2. ``<auth_path>`` (default ``~/.prime/agent/auth.json``) ->
         ``openrouter.key``

    Raises ``OSError`` if neither source yields a non-empty key.  The key is
    never logged or returned in any error message.
    """
    env = os.environ if env is None else env
    key = (env.get("OPENROUTER_API_KEY") or "").strip()
    if key:
        return key

    auth_path = auth_path or DEFAULT_AUTH_PATH
    try:
        if auth_path.is_file():
            data = json.loads(auth_path.read_text(encoding="utf-8"))
            entry = data.get("openrouter")
            if isinstance(entry, dict):
                stored = str(entry.get("key") or "").strip()
                if stored:
                    return stored
    except (OSError, ValueError):
        pass

    raise OSError(
        "OpenRouter API key not found. Set OPENROUTER_API_KEY or provide a "
        f"Prime Agent auth file with an 'openrouter.key' entry ({DEFAULT_AUTH_PATH})."
    )


@dataclass
class OpenRouterTeacherConfig:
    """Static, versioned configuration for the OpenRouter frontier teacher.

    Fields are intentionally const-like so a run is reproducible: changing
    anything here (model, provider pinning, sampling) is a configuration
    change recorded in the run's provenance.
    """

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    provider_only: tuple[str, ...] = DEFAULT_PROVIDER_ONLY
    temperature: float = 0.0
    # None = UNSET (unbound / provider default max output).  Explicitly
    # capping the teacher (e.g. 32) truncates relabelings and full answers;
    # the caller may still override per call.
    max_tokens: int | None = None
    timeout: float = 60.0
    reasoning_enabled: bool = False
    max_retries: int = 3
    seed: int | None = None
    api_key: str | None = None
    auth_path: Path | None = None

    def resolved_api_key(self) -> str:
        return self.api_key or resolve_openrouter_api_key(auth_path=self.auth_path)

    def describe(self) -> dict[str, Any]:
        """Provenance-safe description (no key material)."""
        return {
            "provider": "openrouter",
            "model": self.model,
            "provider_only": list(self.provider_only),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,  # None => unbound/provider default
            "seed": self.seed,
            "reasoning_enabled": self.reasoning_enabled,
        }


class OpenRouterTeacher:
    """Thin OpenAI-compatible chat client pinned to a single OpenRouter model.

    Used as the R18/R19 *frontier teacher*: at evaluation it answers dev
    probes given the correction/fact in context, replacing the 0.5B teacher
    that hit the expressivity ceiling (``teacher_dev_delta=0.1765 < 0.2``).
    """

    def __init__(
        self,
        config: OpenRouterTeacherConfig | None = None,
        *,
        transport: httpx.Transport | None = None,
    ) -> None:
        self.config = config or OpenRouterTeacherConfig()
        self.last_usage: dict[str, Any] | None = None
        self.log: list[dict[str, Any]] = []
        headers = {
            "Authorization": f"Bearer {self.config.resolved_api_key()}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(
            base_url=self.config.base_url,
            headers=headers,
            timeout=self.config.timeout,
            transport=transport,
            follow_redirects=True,
        )

    # ------------------------------------------------------------------
    # Core completion
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> str:
        """Single greedy chat completion; returns the assistant text.

        Raises ``RuntimeError`` with an informative (key-free) message if the
        provider errors persistently.
        """
        resolved_max_tokens = (
            max_tokens if max_tokens is not None else self.config.max_tokens
        )
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "provider": {"only": list(self.config.provider_only)},
            "reasoning": {"enabled": self.config.reasoning_enabled},
        }
        if resolved_max_tokens is not None:
            body["max_tokens"] = resolved_max_tokens
        s = seed if seed is not None else self.config.seed
        if s is not None:
            body["seed"] = s

        last_err: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self._client.post("/chat/completions", json=body)
            except httpx.HTTPError as exc:  # network-level
                last_err = exc
                if attempt < self.config.max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise RuntimeError(f"OpenRouter request failed after retries: {exc}") from exc

            if resp.status_code in _RETRY_STATUSES and attempt < self.config.max_retries:
                last_err = RuntimeError(f"HTTP {resp.status_code}")
                time.sleep(0.5 * (2 ** attempt))
                continue

            if resp.status_code != 200:
                snippet = resp.text[:300]
                raise RuntimeError(
                    f"OpenRouter HTTP {resp.status_code}: {snippet}"
                )

            payload = resp.json()
            self.last_usage = payload.get("usage")
            choices = payload.get("choices") or []
            if not choices:
                raise RuntimeError("OpenRouter returned no choices.")
            content = (choices[0].get("message") or {}).get("content") or ""
            self.log.append({
                "n_messages": len(messages),
                "request_chars": sum(len(m.get("content", "")) for m in messages),
                "usage": self.last_usage,
            })
            return content

        assert last_err is not None
        raise RuntimeError(f"OpenRouter request failed after retries: {last_err}")

    # ------------------------------------------------------------------
    # Probe-oriented helpers (R18 teacher-gate semantics)
    # ------------------------------------------------------------------

    _DEFAULT_SYSTEM = (
        "Consider the definition given in the user message, then respond to "
        "the user's request using that definition in a single short sentence."
    )

    def answer_probe(
        self,
        request: str,
        *,
        correction: str | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Answer a curriculum probe, optionally with the correction in context.

        Mirrors R18's teacher setup: the correction utterance plays the role
        of the local reserved-position prefix and the probe is the request.
        Returns the generated answer text (stripped).
        """
        user = f"{correction}\n\n{request}" if correction else request
        messages = [
            {"role": "system", "content": system or self._DEFAULT_SYSTEM},
            {"role": "user", "content": user},
        ]
        return self.complete(messages, max_tokens=max_tokens or self.config.max_tokens).strip()

    def close(self) -> None:
        self._client.close()


__all__ = [
    "OpenRouterTeacher",
    "OpenRouterTeacherConfig",
    "resolve_openrouter_api_key",
    "DEFAULT_MODEL",
    "DEFAULT_BASE_URL",
    "DEFAULT_PROVIDER_ONLY",
]
