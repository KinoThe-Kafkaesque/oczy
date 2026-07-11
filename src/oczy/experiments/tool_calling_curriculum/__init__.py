"""Code-backed Pi tool-use curriculum data and scoring contracts.

The package deliberately contains no acceptance thresholds. Thresholds remain
inactive until distributions from real-driver baseline and augmented runs are
recorded, as required by the repository's standing agreements.
"""

from .dataset import ToolEpisode, ToolStage, build_tool_curriculum
from .scoring import EpisodeScore, ParsedToolCall, parse_tool_call, score_episode
from .validation import validate_tool_curriculum

__all__ = [
    "EpisodeScore",
    "ParsedToolCall",
    "ToolEpisode",
    "ToolStage",
    "build_tool_curriculum",
    "parse_tool_call",
    "score_episode",
    "validate_tool_curriculum",
]
