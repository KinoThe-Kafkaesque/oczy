#!/usr/bin/env python3
"""Driver that feeds the organism curriculum to an Oczy agent.

Usage:
    python experiments/organism_curriculum/run_curriculum.py
    python experiments/organism_curriculum/run_curriculum.py --agent OrganismAgent
    python experiments/organism_curriculum/run_curriculum.py --lm
    python experiments/organism_curriculum/run_curriculum.py --stages stage_0_grounding stage_1_transfer
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from oczy.experiments.organism import LMBackendAgent, OrganismAgent
from oczy.experiments.organism_curriculum.dataset import Episode, Probe, Stage, build_curriculum
from oczy.experiments.organism_curriculum.scoring import categorize_results, probe_matches
from oczy.experiments.organism_curriculum.validation import validate_curriculum


def _load_real_cortex_agent() -> Any:
    """Load a CortexAgent backed by the local LFM2.5 GGUF model."""
    from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
    from oczy.lm import CVecDriverConfig, LlamaCVecDriver
    from plastic_cortex.kv_cortex import KVCortexConfig

    print("Loading real LlamaCVecDriver...")
    driver = LlamaCVecDriver.load(
        CVecDriverConfig(n_ctx=128, n_threads=4, embedding=True)
    )
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        use_policy_head=True,
        policy_learning_rate=0.001,
    )
    cortex = CortexAgent(cfg, driver=driver)
    cortex.boot()
    print("Real CortexAgent loaded.")
    return cortex


class _DeterministicCortexShim:
    """Lightweight deterministic policy-head stand-in for probe harness.

    Implements the small subset of CortexAgent that OrganismAgent's gated
    policy loop touches: policy_score, policy_update, and predict_value for
    an optional value baseline. No LM is loaded.
    """

    def __init__(self, n_embd: int = 8, d_cortex: int = 4, seed: int = 0) -> None:
        self.n_embd = n_embd
        self.d_cortex = d_cortex
        self.rng = np.random.default_rng(seed)
        self._policy_W = self.rng.standard_normal(n_embd + d_cortex, dtype=np.float64)
        self._policy_b = 0.0
        self._warm = self.rng.standard_normal(d_cortex, dtype=np.float64)
        self._last_hidden = self.rng.standard_normal(n_embd, dtype=np.float64)
        self._prev_hidden = None
        self.world_model_critic = self._ValueShim()

    class _ValueShim:
        def predict_value(self, **kwargs: Any) -> float:
            return 0.5

    def _hidden(self, text: str) -> np.ndarray:
        h = np.zeros(self.n_embd, dtype=np.float64)
        for i, ch in enumerate(text):
            h[i % self.n_embd] += ord(ch) * 0.01
        return h

    def _policy_features(self, candidates: list[str]) -> np.ndarray:
        hidden_matrix = np.asarray(
            [self._hidden(c) for c in candidates], dtype=np.float64
        )
        warm_matrix = np.repeat(self._warm.reshape(1, -1), len(candidates), axis=0)
        return np.hstack([warm_matrix, hidden_matrix])

    def policy_score(self, candidates: list[str]) -> np.ndarray:
        X = self._policy_features(candidates)
        return X @ self._policy_W + self._policy_b

    def policy_update(
        self,
        candidates: list[str],
        chosen_idx: int,
        reward: float,
        baseline: float = 0.0,
    ) -> None:
        scores = self.policy_score(candidates)
        X = self._policy_features(candidates)
        max_score = np.max(scores)
        exps = np.exp(scores - max_score)
        probs = exps / np.sum(exps)
        advantage = reward - baseline
        self._policy_W += 0.05 * advantage * (X[chosen_idx] - probs @ X)
        self._policy_b += 0.05 * advantage * (1.0 - probs[chosen_idx])



class _MockDriver:
    """Deterministic LM driver stand-in for probe harness.

    Returns fixed-shape hidden vectors without loading a model.
    """

    def __init__(self, n_embd: int = 8, n_layers: int = 2) -> None:
        self.n_embd = n_embd
        self.n_layers = n_layers

    def peek_embedding(
        self, text: str, last_token_only: bool = True
    ) -> np.ndarray:
        del last_token_only
        # Deterministic, nearly-orthogonal hidden vectors so the policy head
        # learns a clear corrected-vs-wrong margin in the probe harness.
        idx = sum(ord(c) for c in text) % self.n_embd
        h = np.zeros(self.n_embd, dtype=np.float64)
        h[idx] = 1.0
        # Add a small length signal so repeated text still differs from others.
        h[(idx + 1) % self.n_embd] = float(len(text)) * 0.05
        return h

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        stop: list[str] | str | None = None,
    ) -> str:
        del prompt, max_tokens, temperature, stop
        return "mock"

@dataclass
class EpisodeResult:
    id: str
    initial_request: str
    first_answer: str
    second_answer: str
    corrected_response: str
    fixed: bool
    lm_parse_ok: bool | None = None
    policy_score_before: dict[str, float] | None = None
    policy_score_after: dict[str, float] | None = None


@dataclass
class StageResult:
    name: str
    description: str
    episode_results: list[EpisodeResult] = field(default_factory=list)
    pre_probe_results: list[tuple[Any, str, bool]] = field(default_factory=list)
    post_probe_results: list[tuple[Any, str, bool]] = field(default_factory=list)
    memory_bytes_before: int = 0
    memory_bytes_after: int = 0

    def uptake_latency(self) -> float:
        if not self.episode_results:
            return 0.0
        not_fixed = sum(1 for r in self.episode_results if not r.fixed)
        return not_fixed / len(self.episode_results)


def load_agent(agent_name: str, config: dict[str, Any]) -> OrganismAgent | LMBackendAgent:
    """Construct the requested agent class."""
    if agent_name == "LMBackendAgent":
        return LMBackendAgent(config)
    return OrganismAgent(config)

def _agent_memory_bytes(agent: Any) -> int:
    if hasattr(agent, "memory_bytes"):
        return int(agent.memory_bytes())
    return 0


def _can_instrument_policy(agent: Any) -> bool:
    cortex = getattr(agent, "cortex_agent", None)
    if cortex is None:
        return False
    return callable(getattr(cortex, "policy_score", None))


def _record_policy_scores(agent: Any, candidates: list[str]) -> dict[str, float] | None:
    cortex = getattr(agent, "cortex_agent", None)
    if cortex is None:
        return None
    try:
        scores = cortex.policy_score(candidates)
        return {
            cand: float(scores[i])
            for i, cand in enumerate(candidates)
            if i < len(scores)
        }
    except Exception:
        return None


def run_battery(
    agent: Any,
    stage: Stage,
    episodes: tuple[Episode, ...] | None,
) -> list[tuple[Any, str, bool]]:
    """Run all probes from ``stage`` against ``agent``.

    If ``episodes`` is supplied, only probes belonging to those episodes are
    run (used for pre/post tests scoped to the current stage).
    """
    results: list[tuple[Any, str, bool]] = []
    episode_set = set(episodes) if episodes is not None else None
    for ep in stage.episodes:
        if episode_set is not None and ep not in episode_set:
            continue
        for probe in ep.probes:
            answer = agent.answer(probe.request)
            ok = probe_matches(answer, probe, ep)
            results.append((probe, answer, ok))
    return results


def build_nl_utterance(episode: Episode) -> str:
    """Compose a single natural-language utterance from request + correction."""
    return "%s %s" % (episode.initial_request, episode.correction_utterance)


def run_stage(
    agent: Any,
    stage: Stage,
    adapter: Any | None,
    instrument_policy: bool = False,
) -> StageResult:
    """Present every episode in ``stage`` to ``agent`` and return metrics."""
    result = StageResult(name=stage.name, description=stage.description)
    result.memory_bytes_before = _agent_memory_bytes(agent)

    # Pre-test probes *before* this stage's acquisition episodes.
    result.pre_probe_results = run_battery(agent, stage, stage.episodes)

    for ep in stage.episodes:
        first_answer = agent.answer(ep.initial_request)

        policy_before: dict[str, float] | None = None
        policy_after: dict[str, float] | None = None
        if instrument_policy and _can_instrument_policy(agent):
            candidates = list(getattr(agent.plastic_cortex, "labels", []))
            policy_before = _record_policy_scores(agent, candidates)

        lm_parse_ok: bool | None = None
        if adapter is not None:
            nl = build_nl_utterance(ep)
            parsed = adapter.nl_to_episode(nl)
            parsed_corrected = parsed.get("corrected_answer", "")
            lm_parse_ok = bool(
                parsed_corrected
                and parsed_corrected.lower() in ep.corrected_response.lower()
            )
            query = parsed.get("query") or ep.initial_request
            correction = parsed.get("correction") or ep.correction_utterance
            agent.learn(query, correction)
        else:
            agent.learn(ep.initial_request, ep.correction_utterance)

        second_answer = agent.answer(ep.initial_request)
        if instrument_policy and _can_instrument_policy(agent):
            policy_after = _record_policy_scores(agent, candidates)

        retention_probe = Probe(
            ep.initial_request, ep.corrected_response, "retention", "sense"
        )
        fixed = probe_matches(second_answer, retention_probe, ep)

        result.episode_results.append(
            EpisodeResult(
                id=ep.id,
                initial_request=ep.initial_request,
                first_answer=first_answer,
                second_answer=second_answer,
                corrected_response=ep.corrected_response,
                fixed=fixed,
                lm_parse_ok=lm_parse_ok,
                policy_score_before=policy_before,
                policy_score_after=policy_after,
            )
        )

    # Post-test probes *after* acquisition.
    result.post_probe_results = run_battery(agent, stage, stage.episodes)
    result.memory_bytes_after = _agent_memory_bytes(agent)
    return result


def _shorten(text: str, width: int = 40) -> str:
    text = text.replace("\n", " ")
    if len(text) > width:
        return text[: width - 3] + "..."
    return text


def _accuracy(items: list[bool]) -> float:
    return sum(items) / len(items) if items else 0.0


def _episode_asdict(er: EpisodeResult) -> dict[str, Any]:
    d = asdict(er)
    if d.get("policy_score_before") is None:
        d.pop("policy_score_before", None)
    if d.get("policy_score_after") is None:
        d.pop("policy_score_after", None)
    return d

def print_summary(results: list[StageResult]) -> None:
    header = "%-28s %8s %7s %6s %6s %10s" % (
        "Stage", "Episodes", "Uptake", "Pre", "Post", "Mem d"
    )
    print(header)
    print("-" * len(header))
    for sr in results:
        total = len(sr.episode_results)
        fixed = sum(1 for r in sr.episode_results if r.fixed)
        uptake = sr.uptake_latency()
        pre_acc = categorize_results(sr.pre_probe_results)
        post_acc = categorize_results(sr.post_probe_results)
        pre_total = sum(v[1] for v in pre_acc.values())
        pre_ok = sum(v[0] for v in pre_acc.values())
        post_total = sum(v[1] for v in post_acc.values())
        post_ok = sum(v[0] for v in post_acc.values())
        pre = pre_ok / pre_total if pre_total else 0.0
        post = post_ok / post_total if post_total else 0.0
        mem_delta = sr.memory_bytes_after - sr.memory_bytes_before
        print(
            "%-28s %3d/%-4d %6.2f %5.2f %5.2f %+9dB"
            % (sr.name, fixed, total, uptake, pre, post, mem_delta)
        )


def print_per_stage(results: list[StageResult]) -> None:
    for sr in results:
        print("\n%s" % sr.name)
        if sr.description:
            print("  %s" % sr.description)
        for er in sr.episode_results:
            marker = "OK" if er.fixed else ".."
            lm_info = ""
            if er.lm_parse_ok is not None:
                lm_info = " (lm=%s)" % ("ok" if er.lm_parse_ok else "fail")
            print(
                "  [%s] %-25s 1st=%-25r 2nd=%-25r%s"
                % (
                    marker,
                    er.id,
                    _shorten(er.first_answer, 22),
                    _shorten(er.second_answer, 22),
                    lm_info,
                )
            )
            if er.policy_score_before and er.policy_score_after:
                best_before = max(er.policy_score_before, key=er.policy_score_before.get)
                best_after = max(er.policy_score_after, key=er.policy_score_after.get)
                print(
                    "    policy best: before=%s(%.2f) after=%s(%.2f)"
                    % (
                        best_before,
                        er.policy_score_before[best_before],
                        best_after,
                        er.policy_score_after[best_after],
                    )
                )
        post_acc = categorize_results(sr.post_probe_results)
        if post_acc:
            parts = ", ".join(
                "%s=%.2f" % (cat, acc) for cat, (_ok, _tot, acc) in sorted(post_acc.items())
            )
            print("  post-test accuracy: %s" % parts)


def write_report(
    results: list[StageResult],
    agent_name: str,
    use_lm: bool,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable: list[dict[str, Any]] = []
    for sr in results:
        serializable.append(
            {
                "name": sr.name,
                "description": sr.description,
                "memory_bytes_before": sr.memory_bytes_before,
                "memory_bytes_after": sr.memory_bytes_after,
                "uptake_latency": sr.uptake_latency(),
                "pre_accuracy": {k: v[2] for k, v in categorize_results(sr.pre_probe_results).items()},
                "episodes": [_episode_asdict(er) for er in sr.episode_results],
            }
        )
    payload = {
        "agent": agent_name,
        "use_lm": use_lm,
        "stages": serializable,
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Oczy organism curriculum.")
    p.add_argument(
        "--agent",
        choices=["OrganismAgent", "LMBackendAgent"],
        default="OrganismAgent",
        help="Agent class to evaluate (default: OrganismAgent).",
    )
    p.add_argument(
        "--config",
        default="{}",
        help="JSON config passed to the agent constructor.",
    )
    p.add_argument(
        "--lm",
        action="store_true",
        help="Feed episodes through the LM perception layer (LanguageAdapter).",
    )
    cortex_group = p.add_mutually_exclusive_group()
    cortex_group.add_argument(
        "--use-cortex-shim",
        action="store_true",
        help="Attach a deterministic hand-rolled CortexAgent shim.",
    )
    cortex_group.add_argument(
        "--use-cortex-agent-mock",
        action="store_true",
        help="Attach a CortexAgent with a deterministic mock LM driver.",
    )
    cortex_group.add_argument(
        "--use-real-driver",
        action="store_true",
        help="Attach a CortexAgent backed by the real LFM2.5 GGUF model.",
    )
    p.add_argument(
        "--stages",
        nargs="+",
        help="Run only these stage files by basename (without .json).",
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the curriculum validation smoke test.",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports",
        help="Directory for the JSON report.",
    )
    p.add_argument(
        "--report-name",
        default="run.json",
        help="Report filename.",
    )
    p.add_argument(
        "--policy-log",
        type=Path,
        default=None,
        help="Path to write additional machine-readable policy instrumentation.",
    )
    return p.parse_args(argv)


def _print_policy_delta(results: list[StageResult]) -> None:
    def _match_key(text: str, scores: dict[str, float]) -> str | None:
        """Find the candidate key that corresponds to ``text``.

        First tries an exact lookup, then a case-insensitive containment match,
        then falls back to the key with the greatest token overlap.
        """
        if text in scores:
            return text
        lowered = text.lower()
        for key in scores:
            if key.lower() == lowered:
                return key
            if key.lower() in lowered or lowered in key.lower():
                return key
        text_tokens = set(re.findall(r"[a-z0-9']+", lowered))
        best_key = None
        best_overlap = 0.0
        for key in scores:
            key_tokens = set(re.findall(r"[a-z0-9']+", key.lower()))
            if not key_tokens:
                continue
            overlap = len(text_tokens & key_tokens) / max(len(text_tokens), len(key_tokens))
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = key
        return best_key

    has_scores = any(
        er.policy_score_before and er.policy_score_after
        for sr in results
        for er in sr.episode_results
    )
    if not has_scores:
        return

    absolute_deltas = []
    margin_deltas = []
    for sr in results:
        for er in sr.episode_results:
            if not (er.policy_score_before and er.policy_score_after):
                continue
            before = er.policy_score_before
            after = er.policy_score_after
            corrected = er.corrected_response
            wrong = er.first_answer
            corrected_key = _match_key(corrected, before) or _match_key(corrected, after)
            if corrected_key is None and wrong in before and wrong in after:
                other_keys = [k for k in set(before) | set(after) if k != wrong]
                if len(other_keys) == 1:
                    corrected_key = other_keys[0]

            b_corrected = before.get(corrected_key) if corrected_key else None
            a_corrected = after.get(corrected_key) if corrected_key else None
            if b_corrected is not None and a_corrected is not None:
                absolute_deltas.append(a_corrected - b_corrected)

            b_wrong = before.get(wrong, 0.0)
            a_wrong = after.get(wrong, 0.0)
            b_margin = (before.get(corrected_key, b_wrong) - b_wrong) if corrected_key else 0.0
            a_margin = (after.get(corrected_key, a_wrong) - a_wrong) if corrected_key else 0.0
            margin_deltas.append(a_margin - b_margin)

    print()
    if absolute_deltas:
        avg_absolute = sum(absolute_deltas) / len(absolute_deltas)
        print("Average corrected-answer policy score delta: %.4f" % avg_absolute)
    else:
        print("Policy scores present but corrected-answer keys not found.")
    if margin_deltas:
        avg_margin = sum(margin_deltas) / len(margin_deltas)
        print("Average corrected-answer policy margin delta: %.4f" % avg_margin)
    if not absolute_deltas and not margin_deltas:
        print("Policy scores present but deltas could not be computed.")

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    agent_config: dict[str, Any] = json.loads(args.config)


    if args.use_cortex_shim:
        agent_config.setdefault("use_cortex_policy", True)
        agent_config.setdefault("use_value_baseline", True)
        agent_config.setdefault("use_acceptance_policy_reward", True)
        print("Enabled policy-loop gates for cortex shim.")
    if args.use_cortex_agent_mock:
        agent_config.setdefault("use_cortex_policy", True)
        agent_config.setdefault("use_value_baseline", True)
        agent_config.setdefault("use_acceptance_policy_reward", True)
        print("Enabled policy-loop gates for CortexAgent mock driver.")
    if args.use_real_driver:
        agent_config.setdefault("use_cortex_policy", True)
        agent_config.setdefault("use_value_baseline", True)
        agent_config.setdefault("use_acceptance_policy_reward", True)
        print("Enabled policy-loop gates for real LM driver.")

    stage_names = tuple(args.stages) if args.stages else None
    stages = build_curriculum(stage_names=stage_names)

    if not args.no_validate:
        report = validate_curriculum(stages)
        if not report.ok:
            print("Curriculum validation failed:")
            for e in report.errors:
                print("  - %s" % e)
            return 1
        if report.warnings:
            print("Curriculum validation warnings:")
            for w in report.warnings:
                print("  - %s" % w)

    agent = load_agent(args.agent, agent_config)

    if args.use_cortex_shim and isinstance(agent, OrganismAgent):
        shim = _DeterministicCortexShim()
        if agent.cortex_agent is None:
            agent.cortex_agent = shim
            print("Deterministic CortexAgent shim attached.")
        else:
            print("CortexAgent already present; shim not attached.")

    if args.use_cortex_agent_mock and isinstance(agent, OrganismAgent):
        if agent.cortex_agent is None:
            from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
            from plastic_cortex.kv_cortex import KVCortexConfig

            driver = _MockDriver()
            cfg = CortexAgentConfig(
                cortex=KVCortexConfig(d_cortex=4),
                use_policy_head=True,
            )
            cortex = CortexAgent(cfg, driver=driver)
            cortex.boot()

            # Work around OrganismAgent's array-valued ``_prev_hidden or _last_hidden``
            # guard, which raises on numpy arrays. Masking _prev_hidden lets the
            # value-baseline path fall through to _last_hidden.
            orig_perceive = cortex.perceive

            def _patched_perceive(
                utterance: str, correction_signal: float | None = None
            ) -> np.ndarray:
                out = orig_perceive(utterance, correction_signal=correction_signal)
                cortex._prev_hidden = None
                return out

            cortex.perceive = _patched_perceive.__get__(cortex, CortexAgent)

            agent.cortex_agent = cortex
            print("CortexAgent with mock driver attached.")
        else:
            print("CortexAgent already present; mock driver not attached.")

    if args.use_real_driver and isinstance(agent, OrganismAgent):
        if agent.cortex_agent is None:
            agent.cortex_agent = _load_real_cortex_agent()
        else:
            print("CortexAgent already present; real driver not attached.")

    adapter = None
    if args.lm:
        try:
            from oczy.lm import LanguageAdapter

            adapter = LanguageAdapter()
            adapter.load()
            print("LM perception adapter loaded.")
        except Exception as exc:  # noqa: BLE001
            print("Could not load LM adapter; continuing in raw mode. (%s)" % exc)

    results: list[StageResult] = []
    for stage in stages:
        if stage.consolidate_before:
            print("Consolidating before %s..." % stage.name)
            agent.consolidate()
        print("Running %s..." % stage.name)
        results.append(
            run_stage(
                agent,
                stage,
                adapter,
                instrument_policy=(args.policy_log is not None),
            )
        )
        if stage.consolidate_after:
            print("Consolidating after %s..." % stage.name)
            agent.consolidate()

    print("\n=== Organism curriculum summary ===")
    print_summary(results)
    print_per_stage(results)
    _print_policy_delta(results)

    report_path = args.report_dir / args.report_name
    write_report(results, args.agent, args.lm, report_path)
    print("\nReport written to: %s" % report_path)

    if args.policy_log is not None:
        policy_payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "episodes": [
                {
                    "stage": sr.name,
                    "id": er.id,
                    "policy_score_before": er.policy_score_before,
                    "policy_score_after": er.policy_score_after,
                }
                for sr in results
                for er in sr.episode_results
            ],
        }
        args.policy_log.parent.mkdir(parents=True, exist_ok=True)
        with args.policy_log.open("w", encoding="utf-8") as fh:
            json.dump(policy_payload, fh, indent=2, default=str)
        print("Policy detail written to: %s" % args.policy_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
