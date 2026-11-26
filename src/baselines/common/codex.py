"""Load the repository's shared, API-like ``codex exec`` runtime."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Sequence


def load_codex_core():
    """Return the tested ephemeral Codex backend used by existing baselines."""

    module_name = "_baseline_codex_exec_core"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parents[1] / "paper2poster" / "codex_backend.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared Codex backend from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


core = load_codex_core()
CodexExecError = core.CodexExecError
CodexRuntime = core.CodexRuntime
configure_codex = core.configure_codex
get_codex_run_stats = core.get_codex_run_stats


def invoke_codex(
    prompt: str,
    images: Sequence[Any] = (),
) -> tuple[str, dict[str, int], str | None]:
    """Invoke the configured runtime outside a framework-specific shim.

    Some upstream projects mix their primary agent SDK with direct HTTP VLM
    clients.  This small public bridge keeps those secondary calls on the same
    isolated Codex backend and in the same timing/cost ledger.
    """

    return core._invoke_codex(core._runtime, prompt, images)
