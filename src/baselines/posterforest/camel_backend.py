"""Install the small CAMEL surface used by PosterForest."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.baselines.common.codex import core


class _AnyEnum:
    def __getattr__(self, name: str) -> str:
        return name


class _Config:
    def __init__(self, **kwargs: Any) -> None:
        self.values = kwargs

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)


class BaseMessage:
    """CAMEL-like message carrying text and optional PIL images."""

    @staticmethod
    def make_user_message(
        role_name: str,
        content: str,
        image_list: list[Any] | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        del role_name
        return SimpleNamespace(content=content, image_list=image_list or [])


def install() -> None:
    """Register dependency-light CAMEL modules before importing PosterForest."""

    camel = types.ModuleType("camel")
    agents = types.ModuleType("camel.agents")
    models = types.ModuleType("camel.models")
    messages = types.ModuleType("camel.messages")
    camel_types = types.ModuleType("camel.types")
    configs = types.ModuleType("camel.configs")

    agents.ChatAgent = core.CodexChatAgent
    models.ModelFactory = core.CodexModelFactory
    messages.BaseMessage = BaseMessage
    camel_types.ModelPlatformType = _AnyEnum()
    camel_types.ModelType = _AnyEnum()
    for name in (
        "ChatGPTConfig",
        "QwenConfig",
        "VLLMConfig",
        "OpenRouterConfig",
        "GeminiConfig",
    ):
        setattr(configs, name, _Config)

    camel.agents = agents
    camel.models = models
    camel.messages = messages
    camel.types = camel_types
    camel.configs = configs
    sys.modules.update(
        {
            "camel": camel,
            "camel.agents": agents,
            "camel.models": models,
            "camel.messages": messages,
            "camel.types": camel_types,
            "camel.configs": configs,
        }
    )


def link_runtime_files(work_dir: Path, upstream_root: Path) -> None:
    """Expose prompt files while keeping generated outputs outside third_party."""

    work_dir.mkdir(parents=True, exist_ok=True)
    for name in ("utils",):
        target = work_dir / name
        if not target.exists():
            target.symlink_to(upstream_root / name, target_is_directory=True)
