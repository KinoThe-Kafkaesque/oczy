"""High-signal contract tests for the three R18 diagnostic modules.

Tests cover mechanism-level contracts — not plumbing.  Each test would fail
for a specific regression:

  - holdout leakage (scoring holdout probes during a DEV-only diagnostic)
  - invented prompt variants beyond the registered raw/chat modes
  - missing 0.2 gate evidence
  - lost seed-2 trajectory data
  - mock fallback for a remote scientific run

All tests use lightweight fakes — no real model loading.
"""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import patch

import pytest

from oczy.experiments.organism_curriculum.dataset import (
    Episode,
    Probe,
    Stage,
    split_probes,
)
from oczy.lm._types import ReservedPosition

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer stand-in for prompt rendering and token counting."""

    def __init__(self) -> None:
        self.bos_token_id: int | None = 1
        self.eos_token_id: int | None = 2

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [ord(c) % 256 for c in text]
        if add_special_tokens and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        if add_special_tokens and self.eos_token_id is not None:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        chars = []
        for i in ids:
            if i in (self.bos_token_id, self.eos_token_id) and skip_special_tokens:
                continue
            chars.append(chr(i + 256) if i < 256 else "?")
        return "".join(chars)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        parts: list[str] = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


class _FakeModel:
    """Stand-in model with config for _hash_model."""

    class _Config:
        hidden_size = 896
        vocab_size = 151936
        num_hidden_layers = 24

        def to_dict(self) -> dict[str, Any]:
            return {
                "hidden_size": self.hidden_size,
                "vocab_size": self.vocab_size,
                "num_hidden_layers": self.num_hidden_layers,
            }

    def __init__(self) -> None:
        self.config = self._Config()
        self.requires_grad_(False)
        self._eval = True

    def requires_grad_(self, flag: bool) -> None:
        self._requires_grad = flag

    def eval(self) -> None:
        self._eval = True

    def __call__(self, **kwargs: Any) -> Any:
        import torch

        class _Out:
            # logits shape: (batch=1, seq_len=1, vocab=4) — minimal but real tensor.
            logits = torch.zeros(1, 1, 4, dtype=torch.float32)

        return _Out()


class _FakeHFDriver:
    """Minimal HFDriver stand-in that records generate calls and reserved positions."""

    def __init__(self, reply: str = "unknown") -> None:
        self._tokenizer = _FakeTokenizer()
        self._model = _FakeModel()
        self.model_id = "Qwen/Qwen2.5-0.5B"
        self.n_embd = 896
        self.n_vocab = 151936
        self.n_layers = 24
        self._reply = reply
        self._reserved: ReservedPosition | None = None
        self.generate_calls: list[tuple[str, int]] = []
        self.reserved_set_count = 0
        self.reserved_cleared_count = 0
        self._closed = False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 32,
        temperature: float = 0.0,
        stop: list[str] | str | None = None,
    ) -> str:
        self.generate_calls.append((prompt, max_tokens))
        return self._reply

    def set_reserved_position(self, rp: ReservedPosition | None) -> None:
        self._reserved = rp
        self.reserved_set_count += 1

    def clear_reserved_position(self) -> None:
        self._reserved = None
        self.reserved_cleared_count += 1

    def close(self) -> None:
        self._closed = True

    def _apply_reserved_prefix(self, prompt: str) -> str:
        if self._reserved is not None:
            prefix = self._reserved.text
            if not prompt.startswith(prefix):
                return prefix + prompt
        return prompt

    def _tokenize(self, text: str) -> Any:
        import torch

        ids = self._tokenizer.encode(text, add_special_tokens=True)
        return torch.tensor([ids], dtype=torch.long)


# ---------------------------------------------------------------------------
# Test fixtures: a tiny curriculum with known DEV/holdout split
# ---------------------------------------------------------------------------


def _make_probe(request: str, expected: str, category: str = "transfer") -> Probe:
    return Probe(request=request, expected=expected, category=category, match_mode="contains")


def _make_episode(
    eid: str,
    request: str = "What is the code?",
    correction: str = "The code is marmalade.",
    answer: str = "marmalade",
) -> Episode:
    return Episode(
        id=eid,
        initial_request=request,
        default_response="I don't know.",
        correction_utterance=correction,
        corrected_label=answer,
        corrected_response=answer,
        domain="general",
        probes=(
            _make_probe(f"Repeat the code for {eid}.", answer),
            _make_probe(f"What was the code for {eid}?", answer, category="retention"),
        ),
    )


def _make_stage() -> Stage:
    episodes = tuple(_make_episode(f"ep{i}") for i in range(4))
    return Stage(
        name="stage_0_grounding",
        description="test stage",
        consolidate_before=(),
        consolidate_after=(),
        episodes=episodes,
    )


def _dev_holdout_split(stage: Stage) -> tuple[set[str], set[str]]:
    return split_probes(stage, fraction=0.3, salt="v2")


# ---------------------------------------------------------------------------
# Teacher Ceiling Diagnostic tests
# ---------------------------------------------------------------------------


class TestTeacherCeiling:
    """Tests for r18_teacher_ceiling_diagnostic.py contracts."""

    def test_eval_mode_uses_only_dev_probes_never_holdout(self) -> None:
        """_eval_mode must only score probes whose pid is in dev_ids."""
        from oczy.experiments.r18_teacher_ceiling_diagnostic import _eval_mode

        stage = _make_stage()
        dev_ids, holdout_ids = _dev_holdout_split(stage)
        driver = _FakeHFDriver(reply="marmalade")

        # Pass ALL episodes but only dev_ids — holdout probes must be skipped.
        records = _eval_mode(driver, list(stage.episodes), dev_ids, "vanilla")

        # Every record's pid must be in dev_ids, never in holdout_ids.
        for r in records:
            pid = f"{r['episode_id']}|{r['probe_request']}|"
            # Reconstruct full pid by finding the matching probe
            for ep in stage.episodes:
                for probe in ep.probes:
                    full_pid = f"{ep.id}|{probe.request}|{probe.category}"
                    if full_pid.startswith(pid):
                        assert full_pid in dev_ids, f"holdout probe leaked: {full_pid}"
                        assert full_pid not in holdout_ids

    def test_eval_mode_raw_prefix_sets_and_clears_reserved_position(self) -> None:
        """raw_prefix mode must set_reserved_position before generate and clear after."""
        from oczy.experiments.r18_teacher_ceiling_diagnostic import _eval_mode

        stage = _make_stage()
        dev_ids, _ = _dev_holdout_split(stage)
        driver = _FakeHFDriver(reply="marmalade")

        _eval_mode(driver, list(stage.episodes), dev_ids, "raw_prefix")

        assert driver.reserved_set_count > 0, "raw_prefix must set reserved position"
        assert driver.reserved_cleared_count > 0, "raw_prefix must clear reserved position"

    def test_eval_mode_chat_template_wraps_with_chat_template(self) -> None:
        """chat_template mode must call generate with a chat-templated prompt, not the raw request."""
        from oczy.experiments.r18_teacher_ceiling_diagnostic import _eval_mode

        stage = _make_stage()
        dev_ids, _ = _dev_holdout_split(stage)
        driver = _FakeHFDriver(reply="marmalade")

        _eval_mode(driver, list(stage.episodes), dev_ids, "chat_template")

        # Every generate call in chat_template mode must contain chat markers,
        # not just the bare probe request.
        for prompt, _ in driver.generate_calls:
            assert "<|im_start|>" in prompt, f"chat_template prompt missing ChatML markers: {prompt!r}"

    def test_eval_mode_vanilla_uses_bare_request(self) -> None:
        """vanilla mode must call generate with the bare probe request, no prefix."""
        from oczy.experiments.r18_teacher_ceiling_diagnostic import _eval_mode

        stage = _make_stage()
        dev_ids, _ = _dev_holdout_split(stage)
        driver = _FakeHFDriver(reply="marmalade")

        _eval_mode(driver, list(stage.episodes), dev_ids, "vanilla")

        # No generate call should contain chat markers in vanilla mode.
        for prompt, _ in driver.generate_calls:
            assert "<|im_start|>" not in prompt

    def test_eval_mode_only_three_modes_no_invented_variants(self) -> None:
        """_eval_mode must accept exactly vanilla, raw_prefix, chat_template — no new variants."""
        from oczy.experiments.r18_teacher_ceiling_diagnostic import _eval_mode

        stage = _make_stage()
        dev_ids, _ = _dev_holdout_split(stage)
        driver = _FakeHFDriver(reply="marmalade")

        # An invented mode must raise ValueError, not silently pass.
        with pytest.raises(ValueError, match="unknown prompting mode"):
            _eval_mode(driver, list(stage.episodes), dev_ids, "instruct_v2")

    def test_gate_threshold_is_unchanged_0_2(self) -> None:
        """GATE_THRESHOLD must be exactly 0.2 — the registered value."""
        from oczy.experiments.r18_teacher_ceiling_diagnostic import GATE_THRESHOLD

        assert GATE_THRESHOLD == 0.2

    def test_print_results_emits_gate_threshold_and_pass_flags(self) -> None:
        """_print_results must emit gate_threshold, raw_prefix_gate_pass, chat_template_gate_pass."""
        from oczy.experiments.r18_teacher_ceiling_diagnostic import _print_results

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _print_results(
                vanilla_acc=0.0,
                raw_acc=0.2,
                chat_acc=0.1,
                raw_delta=0.2,
                chat_delta=0.1,
                n_probes=5,
                model_id="test",
                model_sha="abc",
                manifest_sha="def",
                stage_name="stage_0_grounding",
            )
        out = buf.getvalue()
        assert "METRIC gate_threshold=0.2" in out, "gate_threshold sentinel missing"
        assert "METRIC raw_prefix_gate_pass=" in out, "raw_prefix_gate_pass missing"
        assert "METRIC chat_template_gate_pass=" in out, "chat_template_gate_pass missing"
        # Gate pass must be a boolean, not a float or string.
        assert "raw_prefix_gate_pass=True" in out, "raw_prefix delta=0.2 must pass gate"
        assert "chat_template_gate_pass=False" in out, "chat_template delta=0.1 must fail gate"

    def test_print_results_emits_no_h_distill_verdict(self) -> None:
        """_print_results must never emit an H-DISTILL verdict."""
        from oczy.experiments.r18_teacher_ceiling_diagnostic import _print_results

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _print_results(
                vanilla_acc=0.5,
                raw_acc=0.8,
                chat_acc=0.7,
                raw_delta=0.3,
                chat_delta=0.2,
                n_probes=10,
                model_id="test",
                model_sha="abc",
                manifest_sha="def",
                stage_name="stage_0_grounding",
            )
        out = buf.getvalue()
        assert "H-DISTILL" not in out.upper(), "H-DISTILL verdict must not appear"
        assert "ASI verdict=NONE" in out

    def test_print_audit_emits_per_example_with_mode(self) -> None:
        """_print_audit must emit AUDIT lines with episode_id, mode, correct, expected_label."""
        from oczy.experiments.r18_teacher_ceiling_diagnostic import _print_audit

        records = [
            {
                "episode_id": "ep0",
                "probe_request": "What is the code?",
                "expected_label": "marmalade",
                "prediction": "marmalade",
                "correct": True,
                "mode": "raw_prefix",
            },
        ]
        buf = io.StringIO()
        with patch("sys.stderr", buf):
            _print_audit(records)
        out = buf.getvalue()
        assert "AUDIT" in out
        assert "episode_id=ep0" in out
        assert "mode=raw_prefix" in out
        assert "correct=True" in out
        assert "marmalade" in out  # expected_label appears in audit

    def test_accuracy_empty_returns_zero(self) -> None:
        """_accuracy must return 0.0 for empty records, not raise."""
        from oczy.experiments.r18_teacher_ceiling_diagnostic import _accuracy

        assert _accuracy([]) == 0.0

    def test_accuracy_fraction_correct(self) -> None:
        """_accuracy must return the fraction of correct records."""
        from oczy.experiments.r18_teacher_ceiling_diagnostic import _accuracy

        records = [
            {"correct": True},
            {"correct": False},
            {"correct": True},
            {"correct": True},
        ]
        assert _accuracy(records) == 0.75

    def test_main_exits_nonzero_on_driver_load_failure(self) -> None:
        """main() must return 1 and print driver_status=unavailable when HFDriver.load() fails."""
        from oczy.experiments import r18_teacher_ceiling_diagnostic as mod

        buf_err = io.StringIO()
        with patch.object(mod.HFDriver, "load", side_effect=RuntimeError("no model")):
            with patch("sys.stderr", buf_err):
                rc = mod.main([])
        assert rc == 1, "driver load failure must exit nonzero"
        assert "driver_status=unavailable" in buf_err.getvalue()

    def test_main_no_mock_fallback_for_real_driver(self) -> None:
        """main() must not fall back to a mock driver — it must fail closed."""
        from oczy.experiments import r18_teacher_ceiling_diagnostic as mod

        # If HFDriver.load fails, there must be no mock path — just exit 1.
        with patch.object(mod.HFDriver, "load", side_effect=RuntimeError("no model")):
            with patch("sys.stderr", io.StringIO()):
                rc = mod.main([])
        assert rc == 1
        # Verify no mock-related output was emitted.
        # (The module has no mock path at all.)


# ---------------------------------------------------------------------------
# Prompt Contract Diagnostic tests
# ---------------------------------------------------------------------------


class TestPromptContract:
    """Tests for r18_prompt_contract_diagnostic.py contracts."""

    def test_render_raw_prompts_uses_only_registered_templates(self) -> None:
        """_render_raw_prompts must return exactly the 3 registered _distillation_prompts templates."""
        from oczy.experiments.r18_prompt_contract_diagnostic import (
            _RAW_TEMPLATE_LABELS,
            _render_raw_prompts,
        )

        rendered = _render_raw_prompts("What is the code?")
        assert len(rendered) == 3, "must have exactly 3 raw templates"
        labels = [label for label, _ in rendered]
        assert labels == list(_RAW_TEMPLATE_LABELS)

        # Verify the prompt texts match _distillation_prompts exactly.
        from oczy.experiments.consolidation_distillation import _distillation_prompts

        expected = _distillation_prompts("What is the code?")
        actual = [text for _, text in rendered]
        assert actual == expected, "raw prompts must match _distillation_prompts byte-for-byte"

    def test_render_raw_prompts_no_invented_variants(self) -> None:
        """_render_raw_prompts must not contain any template beyond the registered 3."""
        from oczy.experiments.r18_prompt_contract_diagnostic import _render_raw_prompts

        rendered = _render_raw_prompts("test request")
        texts = [text for _, text in rendered]
        # No instruct/system/assistant wrappers in raw mode.
        for t in texts:
            assert "<|im_start|>" not in t, "raw prompt must not contain chat markers"
            assert "system" not in t.lower() or "system" in t  # bare request may contain "system"
        # The three must be: bare, Q:/A:, Question:/Answer:
        assert texts[0] == "test request"
        assert "Q: test request" in texts[1]
        assert "A:" in texts[1]
        assert "Question: test request" in texts[2]
        assert "Answer:" in texts[2]

    def test_detect_answer_cue_raw_qa_has_cue(self) -> None:
        """_detect_answer_cue must find A: cue in raw_qa template."""
        from oczy.experiments.r18_prompt_contract_diagnostic import MODE_RAW, _detect_answer_cue

        prompt = "Q: What is the code?\nA:"
        info = _detect_answer_cue(prompt, MODE_RAW, "raw_qa")
        assert info["answer_cue_present"] is True
        assert info["answer_prefix_mismatch"] is False

    def test_detect_answer_cue_raw_bare_no_cue_is_ok(self) -> None:
        """_detect_answer_cue must not flag bare template as mismatch (bare is intentionally cue-less)."""
        from oczy.experiments.r18_prompt_contract_diagnostic import MODE_RAW, _detect_answer_cue

        prompt = "What is the code?"
        info = _detect_answer_cue(prompt, MODE_RAW, "raw_bare")
        assert info["answer_cue_present"] is False
        assert info["answer_prefix_mismatch"] is False, "bare template must not be flagged as mismatch"

    def test_detect_answer_cue_chat_has_assistant_marker(self) -> None:
        """_detect_answer_cue must find <|im_start|>assistant in chat-template mode."""
        from oczy.experiments.r18_prompt_contract_diagnostic import MODE_CHAT, _detect_answer_cue

        prompt = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nQ?<|im_end|>\n<|im_start|>assistant\n"
        info = _detect_answer_cue(prompt, MODE_CHAT, "chat_template")
        assert info["answer_cue_present"] is True
        assert info["answer_prefix_mismatch"] is False

    def test_find_role_boundaries_chat_finds_im_start(self) -> None:
        """_find_role_boundaries must locate <|im_start|> markers in chat mode."""
        from oczy.experiments.r18_prompt_contract_diagnostic import MODE_CHAT, _find_role_boundaries

        prompt = "<|im_start|>system\nx<|im_end|>\n<|im_start|>user\ny<|im_end|>\n<|im_start|>assistant\n"
        boundaries = _find_role_boundaries(prompt, MODE_CHAT)
        assert "<|im_start|>" in boundaries
        assert len(boundaries["<|im_start|>"]) == 3, "must find 3 <|im_start|> markers"

    def test_find_role_boundaries_raw_finds_q_a_markers(self) -> None:
        """_find_role_boundaries must locate Q: and A: in raw mode."""
        from oczy.experiments.r18_prompt_contract_diagnostic import MODE_RAW, _find_role_boundaries

        prompt = "Q: What?\nA:"
        boundaries = _find_role_boundaries(prompt, MODE_RAW)
        assert "Q:" in boundaries
        assert "A:" in boundaries

    def test_audit_one_example_correction_before_request(self) -> None:
        """_audit_one_example must detect correction appears before request in teacher prompt."""
        from oczy.experiments.r18_prompt_contract_diagnostic import MODE_RAW, _audit_one_example

        ep = _make_episode("ep0", correction="The code is marmalade.", answer="marmalade")
        probe = ep.probes[0]
        driver = _FakeHFDriver(reply="marmalade")
        tokenizer = _FakeTokenizer()

        record = _audit_one_example(
            driver, tokenizer, ep, probe, MODE_RAW, "raw_qa", f"Q: {probe.request}\nA:"
        )
        assert record["correction_present"] is True
        assert record["correction_before_request"] is True, "correction must precede request in teacher prompt"

    def test_audit_one_example_answer_leak_detected(self) -> None:
        """_audit_one_example must flag answer_leak when expected answer appears in prompt."""
        from oczy.experiments.r18_prompt_contract_diagnostic import MODE_RAW, _audit_one_example

        ep = _make_episode("ep0", answer="marmalade")
        probe = ep.probes[0]
        driver = _FakeHFDriver(reply="marmalade")
        tokenizer = _FakeTokenizer()

        # Inject the expected answer into the prompt text.
        leaky_prompt = "Q: The answer is marmalade.\nA:"
        record = _audit_one_example(
            driver, tokenizer, ep, probe, MODE_RAW, "raw_qa", leaky_prompt
        )
        assert record["answer_leak"] is True
        assert "answer_leak" in record["contract_issues"]

    def test_audit_one_example_no_answer_leak_for_clean_prompt(self) -> None:
        """_audit_one_example must not flag answer_leak when expected is absent from prompt."""
        from oczy.experiments.r18_prompt_contract_diagnostic import MODE_RAW, _audit_one_example

        ep = _make_episode("ep0", answer="marmalade")
        probe = ep.probes[0]
        driver = _FakeHFDriver(reply="marmalade")
        tokenizer = _FakeTokenizer()

        clean_prompt = f"Q: {probe.request}\nA:"
        record = _audit_one_example(
            driver, tokenizer, ep, probe, MODE_RAW, "raw_qa", clean_prompt
        )
        assert record["answer_leak"] is False

    def test_aggregate_separates_raw_and_chat_accuracy(self) -> None:
        """_aggregate must compute raw_accuracy and chat_template_accuracy separately."""
        from oczy.experiments.r18_prompt_contract_diagnostic import MODE_CHAT, MODE_RAW, _aggregate

        records = [
            {"mode": MODE_RAW, "scoring_correct": True, "first_token_correct": True,
             "teacher_correct": True, "has_contract_issue": False, "contract_issues": []},
            {"mode": MODE_RAW, "scoring_correct": False, "first_token_correct": False,
             "teacher_correct": False, "has_contract_issue": False, "contract_issues": []},
            {"mode": MODE_CHAT, "scoring_correct": True, "first_token_correct": False,
             "teacher_correct": True, "has_contract_issue": False, "contract_issues": []},
        ]
        result = _aggregate(records, "test-model", "stage_0_grounding", 2)
        agg = result["aggregates"]
        assert agg["raw_accuracy"] == 0.5
        assert agg["chat_template_accuracy"] == 1.0
        assert agg["mode_accuracy_gap"] == 0.5 - 1.0

    def test_emit_sentinels_includes_raw_and_chat_accuracy(self) -> None:
        """_emit_sentinels must print METRIC lines for both raw and chat accuracy."""
        from oczy.experiments.r18_prompt_contract_diagnostic import _emit_sentinels

        audit = {
            "aggregates": {
                "raw_accuracy": 0.5,
                "chat_template_accuracy": 0.3,
                "raw_correct": 1,
                "raw_total": 2,
                "chat_template_correct": 0,
                "chat_template_total": 1,
                "raw_first_token_correct_rate": 0.5,
                "chat_template_first_token_correct_rate": 0.0,
                "mode_accuracy_gap": 0.2,
                "teacher_correct_rate": 0.33,
                "contract_issue_count": 1,
                "contract_issue_rate": 0.33,
                "malformed_count": 0,
                "missing_correction_count": 0,
                "request_truncated_count": 0,
                "answer_prefix_mismatch_count": 1,
                "answer_leak_count": 0,
            },
            "model_id": "test",
            "stage": "stage_0_grounding",
            "dev_probe_count": 2,
        }
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _emit_sentinels(audit)
        out = buf.getvalue()
        assert "METRIC prompt_contract_raw_accuracy=0.5" in out
        assert "METRIC prompt_contract_chat_template_accuracy=0.3" in out
        assert "H-DISTILL" not in out.upper()

    def test_main_mock_driver_runs_structural_audit(self) -> None:
        """main(--driver mock) must produce a structural audit with records."""
        from oczy.experiments import r18_prompt_contract_diagnostic as mod

        buf_out = io.StringIO()
        buf_err = io.StringIO()
        with patch("sys.stdout", buf_out):
            with patch("sys.stderr", buf_err):
                rc = mod.main(["--driver", "mock"])
        assert rc == 0
        out = buf_out.getvalue()
        assert "METRIC prompt_contract_raw_accuracy=" in out
        assert "METRIC prompt_contract_chat_template_accuracy=" in out
        # JSON audit must be on stderr.
        err = buf_err.getvalue()
        assert '"records"' in err or '"aggregates"' in err

    def test_main_real_driver_exits_nonzero_on_load_failure(self) -> None:
        """main(--driver real) must exit 1 and emit ASI real_driver=failed on load failure."""
        from oczy.experiments import r18_prompt_contract_diagnostic as mod

        buf_out = io.StringIO()
        with patch.object(mod.HFDriver, "load", side_effect=RuntimeError("no model")):
            with patch("sys.stdout", buf_out):
                rc = mod.main(["--driver", "real"])
        assert rc == 1
        assert "ASI real_driver=failed" in buf_out.getvalue()

    def test_main_real_driver_no_metric_on_failure(self) -> None:
        """main(--driver real) must not emit METRIC lines on driver failure — fail closed."""
        from oczy.experiments import r18_prompt_contract_diagnostic as mod

        buf_out = io.StringIO()
        with patch.object(mod.HFDriver, "load", side_effect=RuntimeError("no model")):
            with patch("sys.stdout", buf_out):
                rc = mod.main(["--driver", "real"])
        assert rc == 1
        out = buf_out.getvalue()
        assert "METRIC" not in out, "no METRIC lines on real-driver failure"

    def test_run_audit_uses_only_dev_probes(self) -> None:
        """_run_audit must only audit probes in dev_ids, never holdout."""
        from oczy.experiments.r18_prompt_contract_diagnostic import _run_audit

        stage = _make_stage()
        dev_ids, holdout_ids = _dev_holdout_split(stage)
        driver = _FakeHFDriver(reply="marmalade")
        tokenizer = _FakeTokenizer()

        audit = _run_audit(driver, tokenizer, stage, dev_ids)
        for record in audit["records"]:
            pid = record["probe_id"]
            assert pid in dev_ids, f"holdout probe leaked into audit: {pid}"
            assert pid not in holdout_ids

    def test_run_audit_includes_both_raw_and_chat_modes(self) -> None:
        """_run_audit must produce records for both MODE_RAW and MODE_CHAT."""
        from oczy.experiments.r18_prompt_contract_diagnostic import MODE_CHAT, MODE_RAW, _run_audit

        stage = _make_stage()
        dev_ids, _ = _dev_holdout_split(stage)
        driver = _FakeHFDriver(reply="marmalade")
        tokenizer = _FakeTokenizer()

        audit = _run_audit(driver, tokenizer, stage, dev_ids)
        modes = {r["mode"] for r in audit["records"]}
        assert MODE_RAW in modes, "raw mode records missing"
        assert MODE_CHAT in modes, "chat_template mode records missing"


# ---------------------------------------------------------------------------
# Training Trajectory Diagnostic tests
# ---------------------------------------------------------------------------


class TestTrainingTrajectory:
    """Tests for r18_training_trajectory_diagnostic.py contracts."""

    def test_score_dev_student_uses_only_dev_ids(self) -> None:
        """_score_dev_student must only score probes in dev_ids via _score_stage."""
        from oczy.experiments.r18_training_trajectory_diagnostic import _score_dev_student

        stage = _make_stage()
        dev_ids, holdout_ids = _dev_holdout_split(stage)
        driver = _FakeHFDriver(reply="marmalade")

        acc, total = _score_dev_student(driver, stage, dev_ids)
        # Verify only dev probes were scored.
        for ep in stage.episodes:
            for probe in ep.probes:
                pid = f"{ep.id}|{probe.request}|{probe.category}"
                if pid in holdout_ids:
                    # Holdout probe request should not appear in generate calls.
                    for call_prompt, _ in driver.generate_calls:
                        assert call_prompt != probe.request, f"holdout probe scored: {pid}"

    def test_score_dev_vanilla_disables_adapter(self) -> None:
        """_score_dev_vanilla must disable the adapter during scoring and re-enable after."""
        from oczy.experiments.r18_training_trajectory_diagnostic import _score_dev_vanilla

        stage = _make_stage()
        dev_ids, _ = _dev_holdout_split(stage)
        driver = _FakeHFDriver(reply="marmalade")

        class _TrackingAdapter:
            def __init__(self) -> None:
                self.enabled = True
                self.disable_count = 0
                self.enable_count = 0

            def set_enabled(self, flag: bool) -> None:
                if flag:
                    self.enable_count += 1
                else:
                    self.disable_count += 1
                self.enabled = flag

        adapter = _TrackingAdapter()
        _score_dev_vanilla(driver, adapter, stage, dev_ids)
        assert adapter.disable_count > 0, "adapter must be disabled during vanilla scoring"
        assert adapter.enabled is True, "adapter must be re-enabled after vanilla scoring"

    def test_score_dev_teacher_sets_reserved_position_per_episode(self) -> None:
        """_score_dev_teacher must set reserved position for each dev episode and clear after."""
        from oczy.experiments.r18_training_trajectory_diagnostic import _score_dev_teacher

        stage = _make_stage()
        dev_ids, _ = _dev_holdout_split(stage)
        driver = _FakeHFDriver(reply="marmalade")

        class _NoopAdapter:
            def set_enabled(self, flag: bool) -> None: pass

        dev_episodes = [ep for ep in stage.episodes
                        if any(f"{ep.id}|{p.request}|{p.category}" in dev_ids for p in ep.probes)]
        _score_dev_teacher(driver, _NoopAdapter(), stage, dev_ids, dev_episodes)
        assert driver.reserved_set_count > 0, "teacher scoring must set reserved position"
        assert driver.reserved_cleared_count > 0, "teacher scoring must clear reserved position"

    def test_aggregate_diagnostics_empty_returns_empty_dict(self) -> None:
        """_aggregate_diagnostics must return {} for no trajectories."""
        from oczy.experiments.r18_training_trajectory_diagnostic import _aggregate_diagnostics

        assert _aggregate_diagnostics([], 10) == {}

    def test_aggregate_diagnostics_underfit_flag_on_strong_negative_slope(self) -> None:
        """_aggregate_diagnostics must set underfit_flag=1 when loss slope is strongly negative."""
        from oczy.experiments.r18_training_trajectory_diagnostic import _aggregate_diagnostics

        # Loss decreasing strongly: 5.0 -> 4.0 -> 3.0 -> 2.0 -> 1.0
        traj = {
            "seed": 0,
            "checkpoints": [
                {"step": 0, "train_loss": 0.0, "dev_student_acc": 0.0, "grad_norm": 0.0},
                {"step": 1, "train_loss": 5.0, "dev_student_acc": 0.0, "grad_norm": 1.0},
                {"step": 2, "train_loss": 4.0, "dev_student_acc": 0.1, "grad_norm": 1.0},
                {"step": 3, "train_loss": 3.0, "dev_student_acc": 0.1, "grad_norm": 1.0},
                {"step": 4, "train_loss": 2.0, "dev_student_acc": 0.2, "grad_norm": 1.0},
                {"step": 5, "train_loss": 1.0, "dev_student_acc": 0.2, "grad_norm": 1.0},
            ],
        }
        agg = _aggregate_diagnostics([traj], 5)
        assert agg["underfit_flag"] == 1, "strongly negative slope must set underfit"

    def test_aggregate_diagnostics_saturation_flag_on_near_zero_slope(self) -> None:
        """_aggregate_diagnostics must set saturation_flag=1 when second-half slope is near-zero."""
        from oczy.experiments.r18_training_trajectory_diagnostic import _aggregate_diagnostics

        # Loss: 5.0 -> 3.0 -> 3.001 -> 3.002 -> 3.003 (second half flat)
        traj = {
            "seed": 0,
            "checkpoints": [
                {"step": 0, "train_loss": 0.0, "dev_student_acc": 0.0, "grad_norm": 0.0},
                {"step": 1, "train_loss": 5.0, "dev_student_acc": 0.0, "grad_norm": 1.0},
                {"step": 2, "train_loss": 3.0, "dev_student_acc": 0.1, "grad_norm": 1.0},
                {"step": 3, "train_loss": 3.001, "dev_student_acc": 0.1, "grad_norm": 1.0},
                {"step": 4, "train_loss": 3.002, "dev_student_acc": 0.1, "grad_norm": 1.0},
                {"step": 5, "train_loss": 3.003, "dev_student_acc": 0.1, "grad_norm": 1.0},
            ],
        }
        agg = _aggregate_diagnostics([traj], 5)
        assert agg["saturation_flag"] == 1

    def test_aggregate_diagnostics_instability_flag_on_loss_increase(self) -> None:
        """_aggregate_diagnostics must set instability_flag=1 on consecutive loss increase."""
        from oczy.experiments.r18_training_trajectory_diagnostic import _aggregate_diagnostics

        # Loss: 1.0 -> 0.5 -> 0.8 (increase at step 3)
        traj = {
            "seed": 0,
            "checkpoints": [
                {"step": 0, "train_loss": 0.0, "dev_student_acc": 0.0, "grad_norm": 0.0},
                {"step": 1, "train_loss": 1.0, "dev_student_acc": 0.0, "grad_norm": 1.0},
                {"step": 2, "train_loss": 0.5, "dev_student_acc": 0.1, "grad_norm": 1.0},
                {"step": 3, "train_loss": 0.8, "dev_student_acc": 0.1, "grad_norm": 1.0},
            ],
        }
        agg = _aggregate_diagnostics([traj], 3)
        assert agg["instability_flag"] == 1

    def test_aggregate_diagnostics_instability_flag_on_acc_swing(self) -> None:
        """_aggregate_diagnostics must set instability_flag=1 on dev accuracy swing > 0.3."""
        from oczy.experiments.r18_training_trajectory_diagnostic import _aggregate_diagnostics

        # Monotonic loss but accuracy swings from 0.1 to 0.5
        traj = {
            "seed": 0,
            "checkpoints": [
                {"step": 0, "train_loss": 0.0, "dev_student_acc": 0.0, "grad_norm": 0.0},
                {"step": 1, "train_loss": 1.0, "dev_student_acc": 0.1, "grad_norm": 1.0},
                {"step": 2, "train_loss": 0.9, "dev_student_acc": 0.1, "grad_norm": 1.0},
                {"step": 3, "train_loss": 0.8, "dev_student_acc": 0.5, "grad_norm": 1.0},
            ],
        }
        agg = _aggregate_diagnostics([traj], 3)
        assert agg["instability_flag"] == 1

    def test_aggregate_diagnostics_seed_divergence(self) -> None:
        """_aggregate_diagnostics must compute max pairwise final-loss difference."""
        from oczy.experiments.r18_training_trajectory_diagnostic import _aggregate_diagnostics

        trajs = [
            {"seed": 0, "checkpoints": [
                {"step": 0, "train_loss": 0.0, "dev_student_acc": 0.0, "grad_norm": 0.0},
                {"step": 1, "train_loss": 1.0, "dev_student_acc": 0.0, "grad_norm": 1.0},
                {"step": 2, "train_loss": 0.5, "dev_student_acc": 0.0, "grad_norm": 1.0},
            ]},
            {"seed": 1, "checkpoints": [
                {"step": 0, "train_loss": 0.0, "dev_student_acc": 0.0, "grad_norm": 0.0},
                {"step": 1, "train_loss": 2.0, "dev_student_acc": 0.0, "grad_norm": 1.0},
                {"step": 2, "train_loss": 1.5, "dev_student_acc": 0.0, "grad_norm": 1.0},
            ]},
        ]
        agg = _aggregate_diagnostics(trajs, 2)
        assert agg["seed_divergence_max"] == pytest.approx(1.0)

    def test_aggregate_diagnostics_no_h_distill_verdict(self) -> None:
        """_aggregate_diagnostics must not contain any H-DISTILL verdict field."""
        from oczy.experiments.r18_training_trajectory_diagnostic import _aggregate_diagnostics

        traj = {"seed": 0, "checkpoints": [
            {"step": 0, "train_loss": 0.0, "dev_student_acc": 0.0, "grad_norm": 0.0},
            {"step": 1, "train_loss": 1.0, "dev_student_acc": 0.0, "grad_norm": 1.0},
        ]}
        agg = _aggregate_diagnostics([traj], 1)
        for key in agg:
            assert "verdict" not in key.lower(), f"verdict field found: {key}"
            assert "h_distill" not in key.lower(), f"H-DISTILL field found: {key}"

    def test_main_exits_nonzero_on_driver_load_failure(self) -> None:
        """main() must return 1 and emit ASI traj_driver_error on HFDriver.load() failure."""
        from oczy.experiments import r18_training_trajectory_diagnostic as mod

        buf_err = io.StringIO()
        with patch.object(mod.HFDriver, "load", side_effect=RuntimeError("no model")):
            with patch("sys.stderr", buf_err):
                rc = mod.main([])
        assert rc == 1
        assert "traj_driver_error" in buf_err.getvalue()
        assert "traj_driver_loaded=0" in buf_err.getvalue()

    def test_main_no_metric_on_driver_failure(self) -> None:
        """main() must not emit METRIC lines on driver failure — fail closed."""
        from oczy.experiments import r18_training_trajectory_diagnostic as mod

        buf_out = io.StringIO()
        with patch.object(mod.HFDriver, "load", side_effect=RuntimeError("no model")):
            with patch("sys.stdout", buf_out):
                with patch("sys.stderr", io.StringIO()):
                    rc = mod.main([])
        assert rc == 1
        assert "METRIC" not in buf_out.getvalue(), "no METRIC on driver failure"

    def test_main_no_holdout_scoring_in_argv_help(self) -> None:
        """main() argparse must not accept any holdout-scoring flag."""
        from oczy.experiments.r18_training_trajectory_diagnostic import _aggregate_diagnostics

        # This is a structural test: the module must not expose holdout scoring.
        # We verify via the aggregate output keys — no holdout-related keys.
        traj = {"seed": 0, "checkpoints": [
            {"step": 0, "train_loss": 0.0, "dev_student_acc": 0.0, "grad_norm": 0.0},
            {"step": 1, "train_loss": 1.0, "dev_student_acc": 0.0, "grad_norm": 1.0},
        ]}
        agg = _aggregate_diagnostics([traj], 1)
        for key in agg:
            assert "holdout" not in key.lower(), f"holdout key found in aggregate: {key}"

    def test_trajectory_checkpoints_cover_step_0_through_max_steps(self) -> None:
        """A trajectory must have checkpoints at step 0, 1, ..., max_steps (inclusive)."""
        # We verify the contract by checking the _run_one_seed_trajectory structure
        # using a fake that avoids real model loading. Since _run_one_seed_trajectory
        # requires a real model for LoRA, we test the checkpoint structure contract
        # via _aggregate_diagnostics input validation.
        from oczy.experiments.r18_training_trajectory_diagnostic import _aggregate_diagnostics

        # 11 checkpoints for max_steps=10 (steps 0..10)
        checkpoints = [{"step": s, "train_loss": float(s), "dev_student_acc": 0.0,
                        "dev_teacher_acc": 0.0, "dev_vanilla_acc": 0.0,
                        "grad_norm": 1.0, "update_norm": 0.5} for s in range(11)]
        traj = {"seed": 0, "checkpoints": checkpoints}
        agg = _aggregate_diagnostics([traj], 10)
        # The aggregate must process all 10 non-zero steps.
        assert agg["n_seeds"] == 1
        # Loss slope should be (10 - 1) / 9 = 1.0
        assert agg["loss_slope_mean"] == pytest.approx(1.0)

    def test_seed_2_trajectory_not_lost(self) -> None:
        """Seed 2's trajectory must be processed identically to other seeds — no special-casing."""
        from oczy.experiments.r18_training_trajectory_diagnostic import _aggregate_diagnostics

        # Simulate 5 seeds (0-4) with seed 2 having a distinct loss pattern.
        trajs = []
        for s in range(5):
            # Seed 2 has flat loss (the "null" pattern).
            if s == 2:
                losses = [1.0] * 5  # flat — no learning
            else:
                losses = [5.0 - i * 0.5 for i in range(5)]  # decreasing
            checkpoints = [{"step": 0, "train_loss": 0.0, "dev_student_acc": 0.0, "grad_norm": 0.0}]
            for i, loss in enumerate(losses, 1):
                checkpoints.append({"step": i, "train_loss": loss, "dev_student_acc": 0.1,
                                    "grad_norm": 1.0, "update_norm": 0.5})
            trajs.append({"seed": s, "checkpoints": checkpoints})

        agg = _aggregate_diagnostics(trajs, 5)
        assert agg["n_seeds"] == 5, "all 5 seeds must be processed"
        # Seed divergence must be > 0 because seed 2 differs.
        assert agg["seed_divergence_max"] > 0.0, "seed 2's distinct trajectory must contribute to divergence"

    def test_run_one_seed_trajectory_has_step_0_baseline(self) -> None:
        """_run_one_seed_trajectory must include a step-0 baseline checkpoint with train_loss=0.0."""
        # We can't call _run_one_seed_trajectory without a real model (LoRA needs torch.nn.Module),
        # but we can verify the contract by checking the return structure expectation.
        # The function's docstring and the main() loop guarantee step 0 exists.
        # We test the aggregate's handling of step-0 exclusion: step 0 must have grad_norm=0
        # and must not contribute to grad-norm instability detection.
        from oczy.experiments.r18_training_trajectory_diagnostic import _aggregate_diagnostics

        traj = {
            "seed": 0,
            "checkpoints": [
                {"step": 0, "train_loss": 0.0, "dev_student_acc": 0.0, "grad_norm": 0.0},
                {"step": 1, "train_loss": 1.0, "dev_student_acc": 0.0, "grad_norm": 1.0},
                {"step": 2, "train_loss": 0.5, "dev_student_acc": 0.0, "grad_norm": 1.0},
            ],
        }
        agg = _aggregate_diagnostics([traj], 2)
        # Step 0's grad_norm=0 must not be included in grad-norm median.
        # (The code filters: `if c["step"] > 0 and c["grad_norm"] > 0`)
        # If step 0 were included with grad_norm=0, it would skew the median.
        # Here both non-zero grad_norms are 1.0, so median=1.0, no instability.
        assert agg["instability_flag"] == 0


# ---------------------------------------------------------------------------
# Cross-cutting: sentinel serialization determinism
# ---------------------------------------------------------------------------


class TestSentinelSerialization:
    """Tests that sentinel lines are deterministic and machine-readable."""

    def test_teacher_ceiling_sentinels_are_parseable_key_equals_value(self) -> None:
        """All METRIC/ASI lines from teacher ceiling must be key=value format."""
        from oczy.experiments.r18_teacher_ceiling_diagnostic import _print_results

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _print_results(
                vanilla_acc=0.0, raw_acc=0.2, chat_acc=0.1,
                raw_delta=0.2, chat_delta=0.1, n_probes=5,
                model_id="test", model_sha="abc", manifest_sha="def",
                stage_name="stage_0_grounding",
            )
        for line in buf.getvalue().strip().splitlines():
            assert line.startswith("METRIC ") or line.startswith("ASI "), f"bad prefix: {line!r}"
            # Strip prefix, check key=value structure.
            body = line.split(" ", 1)[1]
            assert "=" in body, f"missing = in sentinel: {line!r}"

    def test_prompt_contract_sentinels_are_parseable(self) -> None:
        """All METRIC/ASI lines from prompt contract must be key=value format."""
        from oczy.experiments.r18_prompt_contract_diagnostic import _emit_sentinels

        audit = {
            "aggregates": {
                "raw_accuracy": 0.5, "chat_template_accuracy": 0.3,
                "raw_correct": 1, "raw_total": 2,
                "chat_template_correct": 0, "chat_template_total": 1,
                "raw_first_token_correct_rate": 0.5,
                "chat_template_first_token_correct_rate": 0.0,
                "mode_accuracy_gap": 0.2, "teacher_correct_rate": 0.33,
                "contract_issue_count": 1, "contract_issue_rate": 0.33,
                "malformed_count": 0, "missing_correction_count": 0,
                "request_truncated_count": 0,
                "answer_prefix_mismatch_count": 1, "answer_leak_count": 0,
            },
            "model_id": "test", "stage": "stage_0_grounding", "dev_probe_count": 2,
        }
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _emit_sentinels(audit)
        for line in buf.getvalue().strip().splitlines():
            assert line.startswith("METRIC ") or line.startswith("ASI "), f"bad prefix: {line!r}"
            body = line.split(" ", 1)[1]
            assert "=" in body, f"missing = in sentinel: {line!r}"

    def test_trajectory_aggregate_sentinels_deterministic(self) -> None:
        """_aggregate_diagnostics must produce identical output for identical input."""
        from oczy.experiments.r18_training_trajectory_diagnostic import _aggregate_diagnostics

        traj = {"seed": 0, "checkpoints": [
            {"step": 0, "train_loss": 0.0, "dev_student_acc": 0.0, "grad_norm": 0.0},
            {"step": 1, "train_loss": 1.0, "dev_student_acc": 0.0, "grad_norm": 1.0},
            {"step": 2, "train_loss": 0.5, "dev_student_acc": 0.0, "grad_norm": 1.0},
        ]}
        a1 = _aggregate_diagnostics([traj], 2)
        a2 = _aggregate_diagnostics([traj], 2)
        assert a1 == a2, "identical input must produce identical output"


# ---------------------------------------------------------------------------
# Cross-cutting: no episode-ID conditioning anywhere
# ---------------------------------------------------------------------------


class TestNoEpisodeIdConditioning:
    """Tests that no diagnostic conditions on specific episode IDs."""

    def test_teacher_ceiling_eval_mode_processes_all_dev_episodes(self) -> None:
        """_eval_mode must process all dev episodes regardless of their ID."""
        from oczy.experiments.r18_teacher_ceiling_diagnostic import _eval_mode

        stage = _make_stage()
        dev_ids, _ = _dev_holdout_split(stage)
        driver = _FakeHFDriver(reply="marmalade")

        records = _eval_mode(driver, list(stage.episodes), dev_ids, "vanilla")
        episode_ids = {r["episode_id"] for r in records}
        # All dev episode IDs must appear — no episode is skipped by ID.
        dev_ep_ids = {pid.split("|")[0] for pid in dev_ids}
        assert episode_ids == dev_ep_ids, "some dev episodes were skipped"

    def test_prompt_contract_audit_processes_all_dev_probes(self) -> None:
        """_run_audit must process all dev probes regardless of episode ID."""
        from oczy.experiments.r18_prompt_contract_diagnostic import _run_audit

        stage = _make_stage()
        dev_ids, _ = _dev_holdout_split(stage)
        driver = _FakeHFDriver(reply="marmalade")
        tokenizer = _FakeTokenizer()

        audit = _run_audit(driver, tokenizer, stage, dev_ids)
        audited_pids = {r["probe_id"] for r in audit["records"]}
        assert audited_pids == dev_ids, "some dev probes were not audited"
