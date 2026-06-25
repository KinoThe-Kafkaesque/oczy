"""DigestiveGate: a scalar metabolic gating organ for the Oczy organism.

The gate ingests surprise/novelty/correction signals and produces per-organ
update weights plus a consolidation-pressure scalar. It is intentionally
lightweight (pure Python, no tensor dependencies) so it can run upstream of
every plastic organ without adding a compute budget.

This module is designed to sit alongside the existing ``CortexAgent``
metabolism in ``experiments/cortex_agent.py``; it does not change that
contract or import any of the heavier organ modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DigestiveGateConfig:
    """Tuning knobs for how experience gets metabolised into organ updates."""

    novelty_threshold: float = 0.5
    consolidation_pressure_threshold: float = 0.25
    ema_decay: float = 0.9
    correction_boost: float = 1.0
    immune_suppress_identity: bool = True
    autoencoder_min_weight: float = 0.1


class DigestiveGate:
    """Stateful scalar gate that decides what and how to metabolise."""

    def __init__(self, config: DigestiveGateConfig | None = None) -> None:
        self.config = config or DigestiveGateConfig()
        self._ema: float = 0.0
        self._pressure: float = 0.0

    def _clip(self, value: float) -> float:
        """ Clamp a scalar to the unit interval. """
        return max(0.0, min(1.0, float(value)))

    def ingest(
        self,
        drift: float,
        correction_signal: float,
        novelty: float = 1.0,
        identity_relevance: float = 0.5,
        immune_conflict: float = 0.0,
    ) -> dict[str, Any]:
        """Compute organ weights for a single metabolic step.

        Arguments:
            drift: cortex/latent drift or prediction error, in [0, 1].
            correction_signal: external correction strength, in [0, 1].
            novelty: raw novelty signal, in [0, 1].
            identity_relevance: how relevant this experience is to identity.
            immune_conflict: immune-system override/conflict signal, in [0, 1].

        Returns:
            dict with per-organ float weights and ``consolidation_pressure``.
        """
        cfg = self.config

        drift = self._clip(drift)
        correction_signal = self._clip(correction_signal)
        novelty = self._clip(novelty)
        identity_relevance = self._clip(identity_relevance)
        immune_conflict = self._clip(immune_conflict)

        # correction_boost scales the raw correction signal before any
        # threshold-based gating decisions or consolidation accumulation.
        effective_correction = self._clip(correction_signal * cfg.correction_boost)
        consolidation_input = self._clip(max(drift, effective_correction))

        self._ema = self._clip(
            cfg.ema_decay * self._ema + (1.0 - cfg.ema_decay) * consolidation_input
        )
        # Pressure saturates at the configured threshold -- a deliberately
        # conservative metabolic signal.
        self._pressure = min(self._ema, cfg.consolidation_pressure_threshold)

        # Per-organ heuristics.
        hippocampus_weight = 1.0 if (
            drift > cfg.novelty_threshold and novelty > 0.5
        ) else 0.0

        identity_eligible = effective_correction > 0.5
        identity_suppressed = cfg.immune_suppress_identity and immune_conflict > 0.5
        identity_weight = identity_relevance if (identity_eligible and not identity_suppressed) else 0.0

        immune_weight = 1.0 if (
            effective_correction > 0.5 or immune_conflict > 0.5
        ) else 0.0

        autoencoder_weight = min(
            1.0, cfg.autoencoder_min_weight + max(drift, correction_signal)
        )

        return {
            "critic_weight": 1.0,
            "hippocampus_weight": hippocampus_weight,
            "identity_weight": identity_weight,
            "immune_weight": immune_weight,
            "autoencoder_weight": autoencoder_weight,
            "consolidation_pressure": self._pressure,
        }

    def should_consolidate(self, pressure: float | None = None) -> bool:
        """Return True when consolidation pressure has crossed the threshold."""
        p = self._pressure if pressure is None else self._clip(pressure)
        return p >= self.config.consolidation_pressure_threshold

    def reset(self) -> None:
        """Zero the consolidation-pressure accumulator."""
        self._ema = 0.0
        self._pressure = 0.0
