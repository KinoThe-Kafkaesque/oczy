"""Deterministic regression tests for the OpenRouter frontier-teacher client.

No network, no real key, no provider.  httpx.MockTransport replaces the
wire, and a fake auth file / env var provides the key.  Covers: key
resolution order, provider pinning payload, model slug, auth header,
message shaping for probes with/without correction, greedy sampling
config, retry-and-backoff on transient statuses, and error surfaces.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from oczy.lm.openrouter_teacher import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER_ONLY,
    OpenRouterTeacher,
    OpenRouterTeacherConfig,
    resolve_openrouter_api_key,
)


def _config(api_key: str = "test-key", **kw) -> OpenRouterTeacherConfig:
    return OpenRouterTeacherConfig(api_key=api_key, max_tokens=8, **kw)


def _fake_key_auth(tmp_path: Path, key: str = "auth-file-key") -> Path:
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"openrouter": {"type": "api_key", "key": key}}))
    return auth


# ---------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------


def test_key_from_env():
    env = {"OPENROUTER_API_KEY": "env-key"}
    assert resolve_openrouter_api_key(env=env) == "env-key"


def test_key_falls_back_to_prime_auth_file(tmp_path):
    auth = _fake_key_auth(tmp_path)
    assert resolve_openrouter_api_key(env={}, auth_path=auth) == "auth-file-key"


def test_key_env_wins_over_auth_file(tmp_path):
    auth = _fake_key_auth(tmp_path, key="auth-file-key")
    assert (
        resolve_openrouter_api_key(env={"OPENROUTER_API_KEY": "env-key"}, auth_path=auth)
        == "env-key"
    )


def test_missing_key_raises(tmp_path):
    with pytest.raises(OSError):
        resolve_openrouter_api_key(env={}, auth_path=tmp_path / "does-not-exist.json")


# ---------------------------------------------------------------------
# Wiring: request shape, provider pinning, auth header
# ---------------------------------------------------------------------


def test_request_pins_provider_and_model():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization", "")
        body = json.loads(request.content)
        captured["body"] = body
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "captain\'s journal"}}], "usage": {"total_tokens": 5}},
        )

    transport = httpx.MockTransport(handler)
    teacher = OpenRouterTeacher(_config(), transport=transport)
    out = teacher.answer_probe("Show the log.", correction="No, 'log' means the captain's journal.")

    assert out == "captain's journal"
    assert captured["url"].endswith("/chat/completions")
    assert captured["auth"] == "Bearer test-key"
    body = captured["body"]
    assert body["model"] == DEFAULT_MODEL
    assert body["provider"] == {"only": list(DEFAULT_PROVIDER_ONLY)}
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 8
    # DeepSeek V4-Flash runs a reasoning pass by default; proprietary
    # reasoning burns the token budget and leaves content empty.  The teacher
    # must suppress it for deterministic, content-only probe answers.
    assert body["reasoning"] == {"enabled": False}


def test_answer_probe_message_shaping():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["messages"] = json.loads(request.content)["messages"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    teacher = OpenRouterTeacher(_config(), transport=httpx.MockTransport(handler))

    teacher.answer_probe("Show the log.", correction="No, 'log' means the captain's journal.")
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][1]["role"] == "user"
    assert captured["messages"][1]["content"] == (
        "No, 'log' means the captain's journal.\n\nShow the log."
    )

    teacher.answer_probe("Show the log.", correction=None)
    assert captured["messages"][1]["content"] == "Show the log."


def test_seed_is_forwarded_when_set():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["seed"] = json.loads(request.content).get("seed")
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    teacher = OpenRouterTeacher(
        _config(seed=1234), transport=httpx.MockTransport(handler)
    )
    teacher.answer_probe("p")
    assert captured["seed"] == 1234


def test_unbound_max_tokens_omits_field():
    """max_tokens=None must omit the field entirely (provider default).

    A fixed 32-token cap truncates teacher answers / relabelings; unbound is
    the default.  Passing an explicit value must still be honored.
    """
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "y"}}]})

    # config default is None -> unbound
    teacher = OpenRouterTeacher(
        OpenRouterTeacherConfig(api_key="test-key"),
        transport=httpx.MockTransport(handler),
    )
    teacher.answer_probe("p")
    assert "max_tokens" not in captured["body"]

    # explicit override still honored
    teacher.answer_probe("p", max_tokens=48)
    assert captured["body"]["max_tokens"] == 48


# ---------------------------------------------------------------------
# Retry + error surfaces
# ---------------------------------------------------------------------


def test_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    teacher = OpenRouterTeacher(_config(), transport=httpx.MockTransport(handler))
    assert teacher.answer_probe("p") == "ok"
    assert calls["n"] == 3


def test_non_200_raises_informative_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad auth"}})

    teacher = OpenRouterTeacher(_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="401"):
        teacher.answer_probe("p")


def test_exhausted_retries_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={})

    teacher = OpenRouterTeacher(_config(max_retries=1), transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError):
        teacher.answer_probe("p")
