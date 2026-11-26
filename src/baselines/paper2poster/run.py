"""Run the upstream Paper2Poster pipeline with ``codex exec`` model calls."""

from __future__ import annotations

import argparse
import fcntl
import gc
import importlib
import importlib.metadata
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from codex_backend import (  # type: ignore
        CodexChatAgent,
        CodexModelFactory,
        CodexRuntime,
        configure_codex,
        get_codex_run_stats,
        is_claude_model,
        is_qwen_model,
        load_qwen_config,
    )
else:
    from .codex_backend import (
        CodexChatAgent,
        CodexModelFactory,
        CodexRuntime,
        configure_codex,
        get_codex_run_stats,
        is_claude_model,
        is_qwen_model,
        load_qwen_config,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = REPOSITORY_ROOT / "third_party" / "Paper2Poster"
RELEASE_20250527_UPSTREAM = (
    REPOSITORY_ROOT / "third_party" / "Paper2Poster-20250527"
)
RELEASE_20250527_REVISION = "ae1fbb75e2bd1fb4706c75b224b2864d84a2333e"
RUNTIME_LINKS = {
    "current": ("assets", "config", "logo_store", "utils"),
    "release-20250527": ("assets", "utils"),
}
REPRODUCTION_PROFILES = ("paper-v2", "upstream-current")
PAPER_V2_POSTER_YAML = """# Paper2Poster arXiv v2 compatibility profile.
main_text_font_size: 48
section_title_font_size: 60
poster_title_font_size: 66
poster_author_font_size: 48
title_text_color: [255, 255, 255]
title_fill_color: [47, 85, 151]
main_text_color: null
main_text_fill_color: null
section_title_vertical_align: "top"
section_title_symbol: ""
"""
PAPER_V2_MIN_BODY_SLOT_UNITS = 30.0
_ACTIVE_PARSE_SLOT: tuple[Any, int] | None = None


def _acquire_parse_slot() -> tuple[Any, int] | None:
    raw_slots = os.environ.get("P2P_PAPER2POSTER_PARSE_JOBS")
    raw_root = os.environ.get("P2P_PAPER2POSTER_PARSE_SLOT_DIR")
    if not raw_slots or not raw_root:
        return None
    slots = int(raw_slots)
    if slots < 1:
        raise RuntimeError("P2P_PAPER2POSTER_PARSE_JOBS must be positive")
    root = Path(raw_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    while True:
        for index in range(slots):
            handle = (root / f"slot-{index:02d}.lock").open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            print(
                f"[paper2poster] acquired GPU parse slot {index + 1}/{slots}",
                flush=True,
            )
            return handle, index
        time.sleep(0.25)


def _release_parse_slot() -> None:
    global _ACTIVE_PARSE_SLOT
    if _ACTIVE_PARSE_SLOT is None:
        return
    handle, index = _ACTIVE_PARSE_SLOT
    _ACTIVE_PARSE_SLOT = None
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
    print(f"[paper2poster] released GPU parse slot {index + 1}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an editable poster with Paper2Poster while routing every "
            "LLM/VLM call through the authenticated Codex CLI."
        )
    )
    parser.add_argument("--paper", type=Path, required=True, help="Input paper PDF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty directory for PPTX, previews, traces, and intermediates",
    )
    parser.add_argument(
        "--preparse-cache",
        type=Path,
        help=(
            "GPU-preparsed Markdown and figure/table cache for a generation-only run"
        ),
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=DEFAULT_UPSTREAM,
        help="Paper2Poster checkout (default: third_party/Paper2Poster)",
    )
    parser.add_argument(
        "--upstream-profile",
        choices=tuple(RUNTIME_LINKS),
        default="current",
        help=(
            "Upstream compatibility profile. release-20250527 enforces the "
            "paper-release commit and excludes later theme/runtime behavior."
        ),
    )
    parser.add_argument(
        "--reproduction-profile",
        choices=REPRODUCTION_PROFILES,
        default="paper-v2",
        help=(
            "paper-v2 reproduces the latest paper's documented Painter/theme "
            "settings; upstream-current uses the checkout's later defaults."
        ),
    )
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument(
        "--codex-model",
        default="Qwen3.8-Flash-Next-FP8",
        help="Model passed to Qwen HTTP / claude / codex CLI (default: Qwen3.8-Flash-Next-FP8)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh", "max"),
        default="medium",
        help="Optional reasoning effort override (default: medium)",
    )
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per model call")
    parser.add_argument(
        "--model-retries",
        type=int,
        default=3,
        help="Retries after a failed Codex transport/model call (upstream default: 3)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Concurrent section-generation calls (paper parallel default: 10)",
    )
    parser.add_argument("--poster-width-inches", type=int)
    parser.add_argument("--poster-height-inches", type=int)
    parser.add_argument("--conference-venue")
    parser.add_argument("--institution-logo-path", type=Path)
    parser.add_argument("--conference-logo-path", type=Path)
    parser.add_argument("--use-google-search", action="store_true")
    parser.add_argument("--estimate-chars", action="store_true")
    parser.add_argument("--no-blank-detection", action="store_true")
    parser.add_argument("--no-commenter", action="store_true")
    parser.add_argument("--no-example", action="store_true")
    parser.add_argument("--no-tree-layout", action="store_true")
    parser.add_argument(
        "--no-traces",
        action="store_true",
        help="Do not preserve Codex prompts, JSONL events, and attached images",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep the isolated upstream working directory after success",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing non-empty output directory after generation succeeds",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate paths, executable, and importability without running models",
    )
    return parser


def _validate(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    paper = args.paper.expanduser().resolve()
    upstream = args.upstream_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if args.preparse_cache is not None:
        cache = args.preparse_cache.expanduser().resolve()
        required_cache = (
            cache / "preparse.json",
            cache / "markdown.txt",
            cache / "images_and_tables",
        )
        missing_cache = [str(path) for path in required_cache if not path.exists()]
        if missing_cache:
            raise SystemExit(
                "Incomplete --preparse-cache; missing:\n  "
                + "\n  ".join(missing_cache)
            )
        args.preparse_cache = cache

    if not paper.is_file() or paper.suffix.lower() != ".pdf":
        raise SystemExit(f"--paper must point to an existing PDF: {paper}")
    required = [
        upstream / "PosterAgent" / "new_pipeline.py",
        upstream / "assets" / "poster_data" / "Train",
        upstream / "utils" / "prompt_templates",
    ]
    required.extend(upstream / name for name in RUNTIME_LINKS[args.upstream_profile])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "Paper2Poster checkout is incomplete; missing:\n  " + "\n  ".join(missing)
        )
    if is_qwen_model(args.codex_model):
        load_qwen_config()
    else:
        if is_claude_model(args.codex_model) and args.codex_bin == "codex":
            args.codex_bin = "claude"
        if shutil.which(args.codex_bin) is None:
            raise SystemExit(
                f"Cannot find {args.codex_bin!r} on PATH. Pass --codex-bin if needed."
            )
    if (args.poster_width_inches is None) != (args.poster_height_inches is None):
        raise SystemExit("Set both poster width and height, or neither.")
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be at least 1")
    if args.timeout < 1:
        raise SystemExit("--timeout must be positive")
    if args.model_retries < 0:
        raise SystemExit("--model-retries cannot be negative")
    if args.reproduction_profile == "paper-v2":
        unsupported_visual_options = {
            "--conference-venue": args.conference_venue,
            "--institution-logo-path": args.institution_logo_path,
            "--conference-logo-path": args.conference_logo_path,
            "--use-google-search": args.use_google_search,
        }
        selected_visual_options = [
            flag for flag, value in unsupported_visual_options.items() if value
        ]
        if selected_visual_options:
            raise SystemExit(
                "paper-v2 excludes the post-paper logo extension; use "
                "--reproduction-profile upstream-current for: "
                + ", ".join(selected_visual_options)
            )
    if args.upstream_profile == "release-20250527":
        revision = _git_revision(upstream)
        if revision != RELEASE_20250527_REVISION:
            raise SystemExit(
                "release-20250527 requires Paper2Poster revision "
                f"{RELEASE_20250527_REVISION}, found {revision or 'unknown'} at "
                f"{upstream}"
            )
        unsupported = {
            "--conference-venue": args.conference_venue,
            "--institution-logo-path": args.institution_logo_path,
            "--conference-logo-path": args.conference_logo_path,
            "--use-google-search": args.use_google_search,
            "--estimate-chars": args.estimate_chars,
            "--no-blank-detection": args.no_blank_detection,
        }
        selected = [flag for flag, value in unsupported.items() if value]
        if selected:
            raise SystemExit(
                "The 2025-05-27 release did not support these later options: "
                + ", ".join(selected)
            )
    if output == REPOSITORY_ROOT or output == upstream:
        raise SystemExit("--output-dir cannot be the repository or upstream root")
    if output.exists():
        if not output.is_dir():
            raise SystemExit(f"--output-dir exists but is not a directory: {output}")
        if any(output.iterdir()) and not args.force:
            raise SystemExit(
                f"Output directory is not empty: {output}\n"
                "Choose another directory or explicitly pass --force."
            )
    return paper, upstream, output


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("._")
    return value or "paper"


def _link(source: Path, destination: Path) -> None:
    destination.symlink_to(source, target_is_directory=source.is_dir())


def _prepare_workdir(
    workdir: Path,
    upstream: Path,
    paper: Path,
    slug: str,
    upstream_profile: str,
    reproduction_profile: str,
) -> Path:
    for name in RUNTIME_LINKS[upstream_profile]:
        _link(upstream / name, workdir / name)

    input_dir = workdir / "input" / slug
    input_dir.mkdir(parents=True)
    _link(paper, input_dir / "paper.pdf")
    for sibling_name in ("meta.json", "poster.yaml"):
        sibling = paper.parent / sibling_name
        if sibling.is_file():
            _link(sibling, input_dir / sibling_name)
    if reproduction_profile == "paper-v2":
        # config_utils loads the workdir-level file last, so this deliberately
        # overrides the repository's later default theme.
        (workdir / "poster.yaml").write_text(
            PAPER_V2_POSTER_YAML,
            encoding="utf-8",
        )
    return Path("input") / slug / "paper.pdf"


class _CachedDocument:
    def __init__(self, markdown: str) -> None:
        self._markdown = markdown

    def export_to_markdown(self) -> str:
        return self._markdown


class _CachedConversionResult:
    def __init__(self, markdown: str) -> None:
        self.document = _CachedDocument(markdown)


def _install_preparse_cache(cache: Path, workdir: Path, slug: str) -> dict[str, Any]:
    manifest = json.loads((cache / "preparse.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Preparse cache is not complete: {cache}")
    if manifest.get("slug") != slug:
        raise RuntimeError(
            f"Preparse cache slug mismatch: expected {slug}, found {manifest.get('slug')}"
        )
    markdown = (cache / "markdown.txt").read_text(encoding="utf-8")
    if len(markdown.strip()) < 500:
        raise RuntimeError(f"Preparse Markdown is unexpectedly short: {cache}")
    destination = workdir / "<codex_codex>_images_and_tables"
    shutil.copytree(cache / "images_and_tables", destination)

    parse_raw_module = importlib.import_module("PosterAgent.parse_raw")
    old_converter = getattr(parse_raw_module, "doc_converter", None)
    parse_raw_module.doc_converter = types.SimpleNamespace(
        convert=lambda _source: _CachedConversionResult(markdown)
    )
    del old_converter

    def cached_gen_image_and_table(args: Any, _conversion: Any) -> tuple[Any, ...]:
        root = Path(f"<{args.model_name_t}_{args.model_name_v}>_images_and_tables")
        images = json.loads(
            (root / f"{args.poster_name}_images.json").read_text(encoding="utf-8")
        )
        tables = json.loads(
            (root / f"{args.poster_name}_tables.json").read_text(encoding="utf-8")
        )
        return 0, 0, images, tables

    parse_raw_module.gen_image_and_table = cached_gen_image_and_table
    gc.collect()
    return manifest


def _install_camel_shims() -> None:
    """Expose only the CAMEL symbols used by the upstream pipeline.

    This prevents importing CAMEL provider backends and makes it impossible for
    the adapted path to fall through to an SDK/API call.
    """

    class BaseMessage:
        def __init__(self, content: str, image_list: list | None = None) -> None:
            self.content = content
            self.image_list = image_list

        @classmethod
        def make_user_message(
            cls,
            role_name: str,
            content: str,
            image_list: list | None = None,
            **_: object,
        ) -> "BaseMessage":
            del role_name
            return cls(content=content, image_list=image_list)

    class ModelPlatformType:
        OPENAI_COMPATIBLE_MODEL = "codex-exec"

    class ModelType:
        pass

    class EmptyConfig:
        def as_dict(self) -> dict:
            return {}

    camel = types.ModuleType("camel")
    camel.__path__ = []
    agents = types.ModuleType("camel.agents")
    agents.ChatAgent = CodexChatAgent
    models = types.ModuleType("camel.models")
    models.ModelFactory = CodexModelFactory
    messages = types.ModuleType("camel.messages")
    messages.BaseMessage = BaseMessage
    camel_types = types.ModuleType("camel.types")
    camel_types.ModelPlatformType = ModelPlatformType
    camel_types.ModelType = ModelType
    configs = types.ModuleType("camel.configs")
    for name in (
        "ChatGPTConfig",
        "QwenConfig",
        "VLLMConfig",
        "OpenRouterConfig",
        "GeminiConfig",
    ):
        setattr(configs, name, EmptyConfig)

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


def _install_namespace(name: str, directory: Path) -> None:
    """Install a lightweight package namespace without eager ``__init__``."""

    package = types.ModuleType(name)
    package.__path__ = [str(directory)]
    package.__package__ = name
    sys.modules[name] = package


def _sanitize_bullet_content(bullet_content: Any) -> int:
    """Drop empty model-generated paragraphs that upstream cannot style."""

    removed = 0
    if not isinstance(bullet_content, list):
        return removed
    for section in bullet_content:
        if not isinstance(section, dict):
            continue
        for key in ("title", "textbox1", "textbox2"):
            paragraphs = section.get(key)
            if not isinstance(paragraphs, list):
                continue
            valid = [
                paragraph
                for paragraph in paragraphs
                if not isinstance(paragraph, dict)
                or not isinstance(paragraph.get("runs"), list)
                or bool(paragraph["runs"])
            ]
            removed += len(paragraphs) - len(valid)
            section[key] = valid
    return removed


def _paper_v2_title_height_ratio(
    paper_panels: Sequence[dict[str, Any]],
    base_ratio: float = 0.1,
) -> float:
    """Expand only unusually dense title panels while preserving paper defaults."""

    title_panel = next(
        (
            panel
            for panel in paper_panels
            if "title" in str(panel.get("section_name", "")).lower()
        ),
        None,
    )
    if title_panel is None:
        return base_ratio
    text_length = max(0, int(title_panel.get("text_len", 0)))
    adaptive_ratio = 0.1 + max(0, text_length - 220) / 2200
    return max(base_ratio, min(0.16, adaptive_ratio))


def _rebalance_figure_text_slots(
    panel: dict[str, Any],
    text_boxes: list[dict[str, Any]],
    figure_boxes: list[dict[str, Any]],
    min_body_height: float = PAPER_V2_MIN_BODY_SLOT_UNITS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Center figures below section titles and reserve usable text on both sides."""

    if len(figure_boxes) != 1:
        return text_boxes, figure_boxes
    title_boxes = [box for box in text_boxes if box.get("textbox_id") == 0]
    body_boxes = sorted(
        (box for box in text_boxes if box.get("textbox_id") != 0),
        key=lambda box: int(box.get("textbox_id", 0)),
    )
    if len(title_boxes) != 1 or len(body_boxes) != 2:
        return text_boxes, figure_boxes

    title_box = title_boxes[0]
    top_box, bottom_box = body_boxes
    figure = figure_boxes[0]
    content_top = float(title_box["y"]) + float(title_box["height"])
    content_bottom = float(panel["y"]) + float(panel["height"])
    available_height = max(0.0, content_bottom - content_top)
    body_height = min(float(min_body_height), available_height / 2)
    max_figure_height = max(0.0, available_height - 2 * body_height)

    old_figure_height = float(figure["height"])
    figure_height = min(old_figure_height, max_figure_height)
    if old_figure_height > 0 and figure_height < old_figure_height:
        scale = figure_height / old_figure_height
        old_width = float(figure["width"])
        center_x = float(figure["x"]) + old_width / 2
        new_width = old_width * scale
        figure["x"] = center_x - new_width / 2
        figure["width"] = new_width
        figure["height"] = figure_height

    figure_y = content_top + (available_height - figure_height) / 2
    figure["y"] = figure_y
    top_box["y"] = content_top
    top_box["height"] = max(0.0, figure_y - content_top)
    bottom_box["y"] = figure_y + figure_height
    bottom_box["height"] = max(0.0, content_bottom - bottom_box["y"])
    return text_boxes, figure_boxes


def _adapt_paper_v2_title_fonts(bullet_content: Any) -> dict[str, int]:
    """Keep exact title text while fitting unusually long title metadata."""

    if not isinstance(bullet_content, list) or not bullet_content:
        return {}
    title_section = bullet_content[0]
    if not isinstance(title_section, dict):
        return {}

    def text_length(paragraphs: Any) -> int:
        if not isinstance(paragraphs, list):
            return 0
        return sum(
            len(str(run.get("text", "")))
            for paragraph in paragraphs
            if isinstance(paragraph, dict)
            for run in paragraph.get("runs", [])
            if isinstance(run, dict)
        )

    def cap_font(paragraphs: Any, target: int) -> None:
        if not isinstance(paragraphs, list):
            return
        for paragraph in paragraphs:
            if isinstance(paragraph, dict):
                paragraph["font_size"] = min(
                    int(paragraph.get("font_size", target)),
                    target,
                )

    changes: dict[str, int] = {}
    title_length = text_length(title_section.get("title"))
    if title_length > 150:
        changes["poster_title_font_size"] = 56
    elif title_length > 96:
        changes["poster_title_font_size"] = 60
    author_length = text_length(title_section.get("textbox1"))
    if author_length > 340:
        changes["poster_author_font_size"] = 38
    elif author_length > 220:
        changes["poster_author_font_size"] = 42
    if "poster_title_font_size" in changes:
        cap_font(title_section.get("title"), changes["poster_title_font_size"])
    if "poster_author_font_size" in changes:
        cap_font(title_section.get("textbox1"), changes["poster_author_font_size"])
    return changes


def _install_preparse_import_stubs() -> None:
    """Stub heavy GPU/PDF libraries when running from pre-parsed cache."""
    for mod_name in (
        "docling",
        "docling.datamodel",
        "docling.datamodel.base_models",
        "docling.datamodel.pipeline_options",
        "docling.document_converter",
        "docling_core",
        "docling_core.types",
        "docling_core.types.doc",
        "marker",
        "marker.models",
    ):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    sys.modules["docling_core.types.doc"].ImageRefMode = types.SimpleNamespace(
        EMBEDDED="embedded"
    )
    sys.modules["docling_core.types.doc"].PictureItem = object
    sys.modules["docling_core.types.doc"].TableItem = object
    sys.modules["docling.datamodel.base_models"].InputFormat = types.SimpleNamespace(
        PDF="pdf"
    )
    sys.modules["docling.datamodel.pipeline_options"].PdfPipelineOptions = (
        lambda *a, **kw: types.SimpleNamespace(
            images_scale=5.0,
            generate_page_images=True,
            generate_picture_images=True,
        )
    )
    sys.modules["docling.document_converter"].PdfFormatOption = (
        lambda *a, **kw: None
    )
    sys.modules["docling.document_converter"].DocumentConverter = (
        lambda *a, **kw: None
    )
    sys.modules["marker.models"].create_model_dict = lambda *a, **kw: {}

    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        torch_mod = types.ModuleType("torch")
        torch_mod.float16 = "float16"
        torch_mod.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
        )
        sys.modules["torch"] = torch_mod

    try:
        import playwright.sync_api  # noqa: F401
    except ModuleNotFoundError:
        pw = types.ModuleType("playwright")
        pw_sync = types.ModuleType("playwright.sync_api")
        pw_sync.sync_playwright = None
        pw_async = types.ModuleType("playwright.async_api")
        pw_async.async_playwright = None
        sys.modules["playwright"] = pw
        sys.modules["playwright.sync_api"] = pw_sync
        sys.modules["playwright.async_api"] = pw_async

    try:
        import Levenshtein  # noqa: F401
    except ModuleNotFoundError:
        lev = types.ModuleType("Levenshtein")
        lev.distance = lambda a, b: 0
        sys.modules["Levenshtein"] = lev

    try:
        import json_repair  # noqa: F401
    except ModuleNotFoundError:
        def _repair_loads(s: str, *a: Any, **kw: Any) -> Any:
            s = str(s).strip()
            fence = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL)
            if fence:
                s = fence.group(1).strip()
            s = re.sub(r",\s*([}\]])", r"\1", s)
            return json.loads(s)

        jr = types.ModuleType("json_repair")
        jr.loads = _repair_loads
        sys.modules["json_repair"] = jr

    try:
        import pdf2image  # noqa: F401
    except ModuleNotFoundError:
        from PIL import Image as _PILImage

        def _convert_from_path(pdf_path: str, dpi: int = 72, **_: Any) -> list[Any]:
            with tempfile.TemporaryDirectory(prefix="p2p-pdf2img-") as tmp:
                prefix = Path(tmp) / "page"
                subprocess.run(
                    ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
                    check=True,
                    capture_output=True,
                )
                pages = sorted(Path(tmp).glob("page*.png"))
                result = []
                for p in pages:
                    with _PILImage.open(p) as im:
                        result.append(im.convert("RGB").copy())
                return result

        p2i = types.ModuleType("pdf2image")
        p2i.convert_from_path = _convert_from_path
        sys.modules["pdf2image"] = p2i

    try:
        import sklearn.linear_model  # noqa: F401
    except ModuleNotFoundError:
        import numpy as _np
        from scipy.optimize import minimize as _minimize

        class _LinearRegression:
            def __init__(self, fit_intercept: bool = True, **_: Any) -> None:
                self.fit_intercept = fit_intercept
                self.coef_: Any = None
                self.intercept_: float = 0.0

            def fit(self, X: Any, y: Any) -> "_LinearRegression":
                X_arr = _np.asarray(X, dtype=float)
                y_arr = _np.asarray(y, dtype=float)
                if self.fit_intercept:
                    X_aug = _np.column_stack([X_arr, _np.ones(len(X_arr))])
                    beta, *_ = _np.linalg.lstsq(X_aug, y_arr, rcond=None)
                    self.coef_ = beta[:-1]
                    self.intercept_ = float(beta[-1])
                else:
                    self.coef_, *_ = _np.linalg.lstsq(X_arr, y_arr, rcond=None)
                    self.intercept_ = 0.0
                return self

            def predict(self, X: Any) -> Any:
                X_arr = _np.asarray(X, dtype=float)
                return X_arr @ self.coef_ + self.intercept_

        class _LogisticRegression:
            def __init__(
                self,
                multi_class: str = "multinomial",
                solver: str = "lbfgs",
                fit_intercept: bool = False,
                C: float = 1.0,
                **_: Any,
            ) -> None:
                self.fit_intercept = fit_intercept
                self.C = C
                self.classes_: Any = None
                self.coef_: Any = None

            def fit(self, X: Any, y: Any) -> "_LogisticRegression":
                X_arr = _np.asarray(X, dtype=float)
                y_arr = _np.asarray(y, dtype=int)
                self.classes_ = _np.unique(y_arr)
                n_samples, n_features = X_arr.shape
                n_classes = len(self.classes_)
                Y = _np.zeros((n_samples, n_classes), dtype=float)
                for idx, cls in enumerate(self.classes_):
                    Y[y_arr == cls, idx] = 1.0
                scale = _np.maximum(_np.abs(X_arr).max(axis=0), 1.0)
                X_s = X_arr / scale

                def _loss_grad(w_flat: Any) -> tuple[float, Any]:
                    W = w_flat.reshape(n_classes, n_features)
                    logits = X_s @ W.T
                    logits -= logits.max(axis=1, keepdims=True)
                    exp_l = _np.exp(logits)
                    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
                    loss = -float(_np.sum(Y * _np.log(_np.clip(probs, 1e-15, 1.0))))
                    loss += 0.5 / self.C * float(_np.sum((W / scale) ** 2))
                    grad = (probs - Y).T @ X_s + (W / (scale**2)) / self.C
                    return loss, grad.ravel()

                res = _minimize(
                    _loss_grad,
                    _np.zeros(n_classes * n_features),
                    method="L-BFGS-B",
                    jac=True,
                )
                self.coef_ = res.x.reshape(n_classes, n_features) / scale
                return self

            def predict_proba(self, X: Any) -> Any:
                X_arr = _np.asarray(X, dtype=float)
                logits = X_arr @ self.coef_.T
                logits -= logits.max(axis=1, keepdims=True)
                exp_l = _np.exp(logits)
                return exp_l / exp_l.sum(axis=1, keepdims=True)

            def predict(self, X: Any) -> Any:
                probs = self.predict_proba(X)
                return self.classes_[_np.argmax(probs, axis=1)]

        sk = types.ModuleType("sklearn")
        sk_lm = types.ModuleType("sklearn.linear_model")
        sk_lm.LinearRegression = _LinearRegression
        sk_lm.LogisticRegression = _LogisticRegression
        sk.linear_model = sk_lm
        sys.modules["sklearn"] = sk
        sys.modules["sklearn.linear_model"] = sk_lm


def _patch_upstream(
    upstream: Path,
    upstream_profile: str,
    reproduction_profile: str,
    use_preparse_cache: bool = False,
) -> None:
    # The upstream imports python-dotenv only to load provider API keys. Codex
    # reuses CLI authentication, so keep this dependency optional.
    try:
        importlib.import_module("dotenv")
    except ModuleNotFoundError:
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *args, **kwargs: False
        sys.modules["dotenv"] = dotenv

    if upstream_profile == "release-20250527":
        # The release vendors Docling 2.20.0 source, whose private parser ABI no
        # longer matches modern docling-parse. Preload the installed package so
        # the unchanged release parser uses Docling's stable public API.
        environment_docling = importlib.import_module("docling")
        print(
            "[paper2poster/codex] release parser compatibility: "
            f"environment Docling {importlib.metadata.version('docling')}",
            flush=True,
        )
        del environment_docling

    if use_preparse_cache:
        _install_preparse_import_stubs()
    sys.path.insert(0, str(upstream))
    _install_camel_shims()
    _install_namespace("PosterAgent", upstream / "PosterAgent")
    _install_namespace("utils", upstream / "utils")
    if use_preparse_cache:
        _install_namespace("utils.src", upstream / "utils" / "src")
        model_utils_stub = types.ModuleType("utils.src.model_utils")
        model_utils_stub.parse_pdf = lambda *a, **kw: ("", None)
        sys.modules["utils.src.model_utils"] = model_utils_stub
    for module_name in (
        "PosterAgent.parse_raw",
        "PosterAgent.gen_outline_layout",
        "PosterAgent.gen_poster_content",
        "utils.ablation_utils",
    ):
        module = importlib.import_module(module_name)
        if hasattr(module, "ModelFactory"):
            module.ModelFactory = CodexModelFactory
        if hasattr(module, "ChatAgent"):
            module.ChatAgent = CodexChatAgent

    if upstream_profile == "release-20250527":
        wei_utils = importlib.import_module("utils.wei_utils")
        original_get_agent_config = wei_utils.get_agent_config

        def compatible_get_agent_config(model_name: str) -> dict[str, Any]:
            if model_name == "codex":
                return {
                    "model_type": "codex-exec",
                    "model_platform": "codex-exec",
                    "model_config": {},
                }
            return original_get_agent_config(model_name)

        wei_utils.get_agent_config = compatible_get_agent_config
        return

    if reproduction_profile == "paper-v2":
        tree_layout = importlib.import_module("PosterAgent.tree_split_layout")
        if not getattr(
            tree_layout.generate_constrained_layout,
            "_paper_v2_adaptive_title",
            False,
        ):
            original_generate_constrained_layout = (
                tree_layout.generate_constrained_layout
            )

            def compatible_generate_constrained_layout(
                paper_panels: Sequence[dict[str, Any]],
                poster_w: float,
                poster_h: float,
                title_height_ratio: float = 0.1,
            ) -> Any:
                adaptive_ratio = _paper_v2_title_height_ratio(
                    paper_panels,
                    title_height_ratio,
                )
                return original_generate_constrained_layout(
                    paper_panels,
                    poster_w,
                    poster_h,
                    title_height_ratio=adaptive_ratio,
                )

            setattr(
                compatible_generate_constrained_layout,
                "_paper_v2_adaptive_title",
                True,
            )
            tree_layout.generate_constrained_layout = (
                compatible_generate_constrained_layout
            )

        if not getattr(
            tree_layout.place_text_and_figures_exact,
            "_paper_v2_balanced_figure_slots",
            False,
        ):
            original_place_text_and_figures_exact = (
                tree_layout.place_text_and_figures_exact
            )

            def compatible_place_text_and_figures_exact(
                panel_dict: dict[str, Any],
                figure_model_params: Any,
                section_title_height: float = 32,
            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
                text_boxes, figure_boxes = original_place_text_and_figures_exact(
                    panel_dict,
                    figure_model_params,
                    section_title_height,
                )
                return _rebalance_figure_text_slots(
                    panel_dict,
                    text_boxes,
                    figure_boxes,
                )

            setattr(
                compatible_place_text_and_figures_exact,
                "_paper_v2_balanced_figure_slots",
                True,
            )
            tree_layout.place_text_and_figures_exact = (
                compatible_place_text_and_figures_exact
            )

        parse_raw_module = importlib.import_module("PosterAgent.parse_raw")
        if not getattr(
            parse_raw_module.gen_image_and_table,
            "_releases_gpu_parse_slot",
            False,
        ):
            original_gen_image_and_table = parse_raw_module.gen_image_and_table

            def compatible_gen_image_and_table(*args: Any, **kwargs: Any) -> Any:
                try:
                    return original_gen_image_and_table(*args, **kwargs)
                finally:
                    converter = getattr(parse_raw_module, "doc_converter", None)
                    parse_raw_module.doc_converter = None
                    del converter
                    gc.collect()
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    finally:
                        _release_parse_slot()

            setattr(
                compatible_gen_image_and_table,
                "_releases_gpu_parse_slot",
                True,
            )
            parse_raw_module.gen_image_and_table = compatible_gen_image_and_table

        if not getattr(parse_raw_module.parse_raw, "_preserves_source_order", False):
            original_parse_raw = parse_raw_module.parse_raw

            def ordered_parse_raw(*args: Any, **kwargs: Any) -> Any:
                original_sample = parse_raw_module.random.sample

                def ordered_sample(population: Any, count: int) -> list[Any]:
                    selected = original_sample(population, count)
                    selected_ids = {id(item) for item in selected}
                    return [item for item in population if id(item) in selected_ids]

                parse_raw_module.random.sample = ordered_sample
                try:
                    return original_parse_raw(*args, **kwargs)
                finally:
                    parse_raw_module.random.sample = original_sample

            setattr(ordered_parse_raw, "_preserves_source_order", True)
            parse_raw_module.parse_raw = ordered_parse_raw

        theme_utils = importlib.import_module("utils.theme_utils")
        theme_utils.DEFAULT_THEME["panel_visible"] = True
        theme_utils.DEFAULT_THEME["panel_theme"] = {
            "color": (47, 85, 151),
            "thickness": 5,
            "line_style": "solid",
        }

        # Institution/conference logos were added after the method documented
        # in arXiv v2. Disable automatic discovery for strict reproduction.
        logo_utils = importlib.import_module("utils.logo_utils")
        logo_utils.LogoManager.extract_first_author_institution = (
            lambda self, paper_text: None
        )

    style_utils = importlib.import_module("utils.style_utils")
    if not getattr(style_utils.apply_all_styles, "_codex_compatible", False):
        original_apply_all_styles = style_utils.apply_all_styles

        def compatible_apply_all_styles(
            bullet_content: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            title_changes = (
                _adapt_paper_v2_title_fonts(bullet_content)
                if reproduction_profile == "paper-v2"
                else {}
            )
            if title_changes:
                print(
                    "[paper2poster/codex] adaptive title typography: "
                    + json.dumps(title_changes, sort_keys=True),
                    flush=True,
                )
            removed = _sanitize_bullet_content(bullet_content)
            if removed:
                print(
                    f"[paper2poster/codex] removed {removed} empty bullet paragraph(s) "
                    "before styling",
                    flush=True,
                )
            return original_apply_all_styles(bullet_content, *args, **kwargs)

        setattr(compatible_apply_all_styles, "_codex_compatible", True)
        style_utils.apply_all_styles = compatible_apply_all_styles


def _upstream_argv(
    args: argparse.Namespace,
    relative_paper: Path,
    slug: str,
) -> list[str]:
    result = [
        "PosterAgent.new_pipeline",
        "--poster_path",
        relative_paper.as_posix(),
        "--poster_name",
        slug,
        "--model_name_t",
        "codex",
        "--model_name_v",
        "codex",
        "--tmp_dir",
        "tmp",
    ]
    if args.upstream_profile == "current":
        result.extend(("--max_workers", str(args.max_workers)))
    option_map = [
        ("poster_width_inches", "--poster_width_inches"),
        ("poster_height_inches", "--poster_height_inches"),
    ]
    if args.upstream_profile == "current":
        option_map.append(("conference_venue", "--conference_venue"))
    for attribute, flag in option_map:
        value = getattr(args, attribute)
        if value is not None:
            result.extend((flag, str(value)))
    path_map = []
    if args.upstream_profile == "current":
        path_map.extend(
            (
                ("institution_logo_path", "--institution_logo_path"),
                ("conference_logo_path", "--conference_logo_path"),
            )
        )
    for attribute, flag in path_map:
        value = getattr(args, attribute)
        if value is not None:
            result.extend((flag, str(value.expanduser().resolve())))
    bool_map = [
        ("no_commenter", "--ablation_no_commenter"),
        ("no_example", "--ablation_no_example"),
        ("no_tree_layout", "--ablation_no_tree_layout"),
    ]
    if args.upstream_profile == "current":
        bool_map[0:0] = [
            ("use_google_search", "--use_google_search"),
            ("estimate_chars", "--estimate_chars"),
            ("no_blank_detection", "--no_blank_detection"),
        ]
    for attribute, flag in bool_map:
        if getattr(args, attribute):
            result.append(flag)
    return result


def _git_revision(upstream: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _copy_intermediates(workdir: Path, output: Path) -> None:
    intermediate_root = output / "intermediates"
    for name in ("contents", "tree_splits", "<codex_codex>_images_and_tables"):
        source = workdir / name
        if source.exists():
            shutil.copytree(source, intermediate_root / name)


def _publish(
    args: argparse.Namespace,
    workdir: Path,
    output: Path,
    slug: str,
    upstream: Path,
    paper: Path,
    started_at: str,
    duration_seconds: float,
    codex_usage: dict,
    preparse_manifest: dict[str, Any] | None,
) -> None:
    generated = (
        workdir
        / "<codex_codex>_generated_posters"
        / "input"
        / slug
    )
    if not generated.is_dir():
        raise RuntimeError(f"Paper2Poster did not create its expected output: {generated}")

    if output.exists():
        if any(output.iterdir()):
            if not args.force:
                raise RuntimeError(f"Refusing to replace non-empty output: {output}")
            shutil.rmtree(output)
        else:
            output.rmdir()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(generated), str(output))

    _copy_intermediates(workdir, output)
    trace_dir = workdir / "codex_traces"
    if trace_dir.exists():
        shutil.move(str(trace_dir), str(output / "codex_traces"))

    manifest = {
        "baseline": "Paper2Poster",
        "model_backend": "codex exec",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration_seconds, 2),
        "poster_generation_seconds": round(duration_seconds, 2),
        "gpu_preparse_seconds": (
            round(float(preparse_manifest["gpu_preparse_seconds"]), 2)
            if preparse_manifest is not None
            else None
        ),
        "total_pipeline_seconds": (
            round(
                duration_seconds
                + float(preparse_manifest["gpu_preparse_seconds"]),
                2,
            )
            if preparse_manifest is not None
            else round(duration_seconds, 2)
        ),
        "preparse_cache": (
            str(args.preparse_cache) if args.preparse_cache is not None else None
        ),
        "codex_usage": codex_usage,
        "codex_model": args.codex_model or "Codex default",
        "reasoning_effort": args.reasoning_effort or "Codex default",
        "paper": str(paper),
        "upstream_root": str(upstream),
        "upstream_revision": _git_revision(upstream),
        "upstream_profile": args.upstream_profile,
        "reproduction_profile": args.reproduction_profile,
        "compatibility": {
            "pdf_parser": (
                "environment-docling-public-api"
                if args.upstream_profile == "release-20250527"
                else "upstream-vendored"
            ),
            "docling_version": (
                preparse_manifest.get("docling_version", "cached")
                if preparse_manifest is not None
                else "unknown"
            ),
            "model_transport_only": "codex exec",
            "role_transport": "native-developer-user-assistant-session",
            "project_instructions": "disabled",
            "model_tools": "disabled",
            "model_call_retries": args.model_retries,
            "codex_prompt_compatibility": (
                "overflow-shortening-means-reduce-rendered-lines"
            ),
            "section_selection": (
                "random-subset-preserve-source-order"
                if args.reproduction_profile == "paper-v2"
                else "upstream"
            ),
            "visual_profile": (
                "paper-v2-48pt-body-60pt-title-blue-panel"
                if args.reproduction_profile == "paper-v2"
                else "upstream-current"
            ),
            "layout_contract": (
                "adaptive-title-header-balanced-figure-text-slots"
                if args.reproduction_profile == "paper-v2"
                else "upstream"
            ),
            "title_fit": (
                "exact-text-adaptive-header-and-font-caps"
                if args.reproduction_profile == "paper-v2"
                else "upstream"
            ),
            "post_paper_logo_extension": (
                "disabled"
                if args.reproduction_profile == "paper-v2"
                else "upstream"
            ),
        },
        "max_workers": (
            args.max_workers if args.upstream_profile == "current" else None
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "traces_preserved": not args.no_traces,
    }
    (output / "adapter_run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    global _ACTIVE_PARSE_SLOT
    args = _parser().parse_args(argv)
    paper, upstream, output = _validate(args)
    slug = _slug(paper.parent.name)

    if args.check:
        try:
            _patch_upstream(
                upstream,
                args.upstream_profile,
                args.reproduction_profile,
                use_preparse_cache=args.preparse_cache is not None,
            )
        except ModuleNotFoundError as exc:
            raise SystemExit(
                f"Missing Paper2Poster runtime dependency: {exc.name}\n"
                "Install src/baseline/paper2poster/requirements.txt in the "
                "selected environment."
            ) from exc
        print(f"Paper2Poster: {upstream}")
        print(f"Upstream profile: {args.upstream_profile}")
        print(f"Reproduction profile: {args.reproduction_profile}")
        print(f"Revision: {_git_revision(upstream) or 'unknown'}")
        print(f"Paper: {paper}")
        print(f"Output: {output}")
        print(f"Codex: {shutil.which(args.codex_bin)}")
        print("Adapter paths and upstream imports passed.")
        return 0

    run_started_at = datetime.now(timezone.utc).isoformat()
    run_started_monotonic = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(
        tempfile.mkdtemp(prefix=f".paper2poster-{slug}-", dir=output.parent)
    )
    succeeded = False
    original_cwd = Path.cwd()
    original_argv = sys.argv[:]
    try:
        relative_paper = _prepare_workdir(
            workdir,
            upstream,
            paper,
            slug,
            args.upstream_profile,
            args.reproduction_profile,
        )
        trace_dir = None if args.no_traces else workdir / "codex_traces"
        configure_codex(
            CodexRuntime(
                executable=args.codex_bin,
                model=args.codex_model,
                reasoning_effort=args.reasoning_effort,
                timeout_s=args.timeout,
                max_retries=args.model_retries,
                cwd=workdir,
                trace_dir=trace_dir,
                result_dir=output,
            )
        )
        if args.preparse_cache is None:
            _ACTIVE_PARSE_SLOT = _acquire_parse_slot()
        _patch_upstream(
            upstream,
            args.upstream_profile,
            args.reproduction_profile,
            use_preparse_cache=args.preparse_cache is not None,
        )
        preparse_manifest = (
            _install_preparse_cache(args.preparse_cache, workdir, slug)
            if args.preparse_cache is not None
            else None
        )

        os.chdir(workdir)
        sys.argv = _upstream_argv(args, relative_paper, slug)
        runpy.run_path(
            str(upstream / "PosterAgent" / "new_pipeline.py"),
            run_name="__main__",
        )
        duration_seconds = time.monotonic() - run_started_monotonic
        codex_usage = get_codex_run_stats()
        _publish(
            args,
            workdir,
            output,
            slug,
            upstream,
            paper,
            run_started_at,
            duration_seconds,
            codex_usage,
            preparse_manifest,
        )
        succeeded = True
        print(
            f"Paper2Poster metrics: {duration_seconds:.1f}s, "
            f"calls={codex_usage['call_count']}, "
            f"api-equivalent=${codex_usage['estimated_api_cost_usd']:.4f}"
        )
        print(f"Paper2Poster output: {output}")
        return 0
    finally:
        _release_parse_slot()
        os.chdir(original_cwd)
        sys.argv = original_argv
        if succeeded and not args.keep_workdir:
            shutil.rmtree(workdir)
        else:
            print(f"Paper2Poster workdir retained at: {workdir}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
