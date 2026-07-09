"""Harbor agent package for Oczy vs vanilla LFM benchmark.

Usage:
    harbor run -d <dataset> --agent benchmarks.harbor.agents:OczyAgent
    harbor run -d <dataset> --agent benchmarks.harbor.agents:VanillaAgent
"""

from benchmarks.harbor.agents.oczy_agent import OczyAgent
from benchmarks.harbor.agents.vanilla_agent import VanillaAgent

__all__ = ["OczyAgent", "VanillaAgent"]
