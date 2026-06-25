"""Unit tests for LM config classes: defaults, env overrides, and named profiles.

These tests do not load a model; they only exercise the dataclass constructors and
classmethod helpers.
"""

from __future__ import annotations

import os
from typing import Iterator

import pytest

from oczy.lm import CVecDriverConfig, LanguageAdapterConfig


@pytest.fixture(autouse=True)
def _clean_env() -> Iterator[None]:
    """Clear OCZY_* environment variables before/after each test."""
    prefixes = ("OCZY_",)
    original = {k: v for k, v in os.environ.items() if k.startswith(prefixes)}
    for k in list(os.environ):
        if k.startswith(prefixes):
            del os.environ[k]
    yield
    for k in list(os.environ):
        if k.startswith(prefixes):
            del os.environ[k]
    os.environ.update(original)


class TestCVecDriverConfigDefaults:
    def test_defaults_unchanged(self) -> None:
        cfg = CVecDriverConfig()
        assert cfg.repo_id == "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
        assert cfg.file_name == "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
        assert cfg.n_ctx == 512
        assert cfg.n_threads == 4
        assert cfg.n_gpu_layers == 0
        assert cfg.use_mmap is True
        assert cfg.use_mlock is False
        assert cfg.verbose is False
        assert cfg.embedding is True

    def test_from_env_overrides(self) -> None:
        os.environ["OCZY_N_CTX"] = "2048"
        os.environ["OCZY_N_THREADS"] = "8"
        os.environ["OCZY_N_GPU_LAYERS"] = "12"
        os.environ["OCZY_USE_MMAP"] = "false"
        os.environ["OCZY_USE_MLOCK"] = "1"
        os.environ["OCZY_VERBOSE"] = "yes"
        os.environ["OCZY_MODEL_REPO_ID"] = "custom/repo"
        os.environ["OCZY_MODEL_FILE_NAME"] = "model.gguf"

        cfg = CVecDriverConfig.from_env()
        assert cfg.n_ctx == 2048
        assert cfg.n_threads == 8
        assert cfg.n_gpu_layers == 12
        assert cfg.use_mmap is False
        assert cfg.use_mlock is True
        assert cfg.verbose is True
        assert cfg.repo_id == "custom/repo"
        assert cfg.file_name == "model.gguf"
        assert cfg.embedding is True  # default preserved

    def test_from_env_explicit_overrides_win(self) -> None:
        os.environ["OCZY_N_CTX"] = "2048"
        cfg = CVecDriverConfig.from_env(n_ctx=128, embedding=False)
        assert cfg.n_ctx == 128
        assert cfg.embedding is False

    def test_perception_profile(self) -> None:
        cfg = CVecDriverConfig.perception()
        assert cfg.n_ctx == 1024
        assert cfg.embedding is True
        assert cfg.n_threads == 4  # default preserved

    def test_articulation_profile(self) -> None:
        cfg = CVecDriverConfig.articulation()
        assert cfg.n_ctx == 512
        assert cfg.embedding is False

    def test_benchmark_profile(self) -> None:
        cfg = CVecDriverConfig.benchmark()
        assert cfg.n_ctx == 512
        assert cfg.n_threads == 4
        assert cfg.embedding is True


class TestLanguageAdapterConfigDefaults:
    def test_defaults_unchanged(self) -> None:
        cfg = LanguageAdapterConfig()
        assert cfg.repo_id == "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
        assert cfg.file_name == "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
        assert cfg.n_threads == 4
        assert cfg.n_ctx == 1024
        assert cfg.n_gpu_layers == 0
        assert cfg.use_mmap is True
        assert cfg.use_mlock is False
        assert cfg.verbose is False
        assert cfg.temperature == 0.0
        assert cfg.top_p == 1.0
        assert cfg.max_tokens_parse == 300
        assert cfg.max_tokens_render == 80

    def test_from_env_overrides(self) -> None:
        os.environ["OCZY_TEMPERATURE"] = "0.7"
        os.environ["OCZY_TOP_P"] = "0.9"
        os.environ["OCZY_MAX_TOKENS_PARSE"] = "600"
        os.environ["OCZY_MAX_TOKENS_RENDER"] = "120"
        os.environ["OCZY_N_CTX"] = "2048"
        os.environ["OCZY_USE_MMAP"] = "0"

        cfg = LanguageAdapterConfig.from_env()
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.9
        assert cfg.max_tokens_parse == 600
        assert cfg.max_tokens_render == 120
        assert cfg.n_ctx == 2048
        assert cfg.use_mmap is False

    def test_perception_profile(self) -> None:
        cfg = LanguageAdapterConfig.perception()
        assert cfg.temperature == 0.0
        assert cfg.top_p == 1.0
        assert cfg.max_tokens_parse == 600

    def test_render_profile(self) -> None:
        cfg = LanguageAdapterConfig.render()
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.9
        assert cfg.max_tokens_render == 120

    def test_benchmark_profile(self) -> None:
        cfg = LanguageAdapterConfig.benchmark()
        assert cfg.temperature == 0.0
        assert cfg.top_p == 1.0
        assert cfg.max_tokens_parse == 200
        assert cfg.max_tokens_render == 40


def test_public_acceptance_check() -> None:
    """Acceptance snippet from the task description."""
    c = CVecDriverConfig.perception()
    a = LanguageAdapterConfig.benchmark()
    assert c.n_ctx == 1024
    assert a.temperature == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
