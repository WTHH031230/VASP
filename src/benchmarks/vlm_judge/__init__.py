"""Paper2Poster-compatible VLM judge backed by ``codex exec``."""

from .judge import JudgeConfig, JudgeError, judge_poster
from .rubrics import AESTHETIC_ASPECTS, ALL_ASPECTS, INFORMATION_ASPECTS

__all__ = [
    "AESTHETIC_ASPECTS",
    "ALL_ASPECTS",
    "INFORMATION_ASPECTS",
    "JudgeConfig",
    "JudgeError",
    "judge_poster",
]
