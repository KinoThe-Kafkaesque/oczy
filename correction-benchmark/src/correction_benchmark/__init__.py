"""Public exports for Correction-to-Competence Benchmark."""

from .baseline_agents import AlwaysWrongAgent, OracleAgent
from .benchmark import run_benchmark
from .dataset import Episode, Probe, build_dataset
from .scorer import EpisodeResult, ProbeResult, Scorer

__all__ = [
    "AlwaysWrongAgent",
    "Episode",
    "EpisodeResult",
    "OracleAgent",
    "Probe",
    "ProbeResult",
    "Scorer",
    "build_dataset",
    "run_benchmark",
]
