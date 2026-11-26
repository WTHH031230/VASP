"""Run upstream PosterGen with all LLM/VLM calls routed through Codex Exec."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from codex_backend import (  # type: ignore
        CodexRuntime,
        configure_codex,
        get_codex_run_stats,
        install_langgraph_shim,
        is_claude_model,
        is_qwen_model,
        load_qwen_config,
    )
else:
    from .codex_backend import (
        CodexRuntime,
        configure_codex,
        get_codex_run_stats,
        install_langgraph_shim,
        is_claude_model,
        is_qwen_model,
        load_qwen_config,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = REPOSITORY_ROOT / "third_party" / "PosterGen"
DEFAULT_LEDGER = Path.home() / ".codex" / "usage-ledger" / "ephemeral.jsonl"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--preparse-cache",
        type=Path,
        help="Pre-extracted Marker raw.md and assets/ directory for zero-GPU execution",
    )
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-model", default="Qwen3.8-Flash-Next-FP8")
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh", "max"),
        default="medium",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--parser-lock",
        type=Path,
        default=Path("/tmp/postergen-marker-gpu.lock"),
        help="Cross-process lock that serializes the GPU-heavy Marker stage",
    )
    parser.add_argument("--no-parser-lock", action="store_true")
    parser.add_argument("--poster-width", type=float, default=54.0)
    parser.add_argument("--poster-height", type=float, default=36.0)
    parser.add_argument("--logo", type=Path)
    parser.add_argument("--aff-logo", type=Path)
    parser.add_argument("--url", default="")
    parser.add_argument("--usage-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--no-global-ledger", action="store_true")
    parser.add_argument("--no-traces", action="store_true")
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser


def _validate(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    paper = args.paper.expanduser().resolve()
    upstream = args.upstream_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not paper.is_file():
        raise SystemExit(f"Paper PDF not found: {paper}")
    if paper.suffix.lower() != ".pdf":
        raise SystemExit(f"Input must be a PDF: {paper}")
    if not (upstream / "src" / "workflow" / "pipeline.py").is_file():
        raise SystemExit(f"PosterGen checkout is incomplete: {upstream}")
    ratio = args.poster_width / args.poster_height
    if not 1.4 <= ratio <= 2.0:
        raise SystemExit(f"Poster ratio must be between 1.4 and 2.0, got {ratio:.3f}")
    for label, value in (("logo", args.logo), ("affiliation logo", args.aff_logo)):
        if value is not None and not value.expanduser().resolve().is_file():
            raise SystemExit(f"{label.title()} not found: {value}")
    if output.exists() and any(output.iterdir()) and not args.force:
        raise SystemExit(f"Refusing to replace non-empty output: {output}")
    if args.preparse_cache is not None:
        cache = args.preparse_cache.expanduser().resolve()
        required_cache = (
            cache / "raw.md",
            cache / "assets" / "figures.json",
            cache / "assets" / "tables.json",
        )
        missing_cache = [str(p) for p in required_cache if not p.is_file()]
        if missing_cache:
            raise SystemExit(
                "Incomplete --preparse-cache for PosterGen; missing:\n  "
                + "\n  ".join(missing_cache)
            )
        args.preparse_cache = cache
    if is_qwen_model(args.codex_model):
        load_qwen_config()
    else:
        if is_claude_model(args.codex_model) and args.codex_bin == "codex":
            args.codex_bin = "claude"
        if shutil.which(args.codex_bin) is None:
            raise SystemExit(f"CLI executable not found: {args.codex_bin}")
    return paper, upstream, output


def _install_marker_stubs() -> None:
    """Stub Marker and dotenv modules when running from pre-parsed GPU cache."""
    import types

    try:
        import dotenv  # noqa: F401
    except ModuleNotFoundError:
        dotenv_mod = types.ModuleType("dotenv")
        dotenv_mod.load_dotenv = lambda *a, **kw: False
        sys.modules["dotenv"] = dotenv_mod

    try:
        import qrcode  # noqa: F401
    except ModuleNotFoundError:
        qr_mod = types.ModuleType("qrcode")
        qr_mod.constants = types.SimpleNamespace(ERROR_CORRECT_L=1)
        sys.modules["qrcode"] = qr_mod

    for mod_name in (
        "marker",
        "marker.converters",
        "marker.converters.pdf",
        "marker.renderers",
        "marker.renderers.markdown",
        "marker.models",
        "marker.output",
        "marker.schema",
    ):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    sys.modules["marker.converters.pdf"].PdfConverter = lambda *a, **kw: None
    sys.modules["marker.renderers.markdown"].MarkdownRenderer = object
    sys.modules["marker.models"].create_model_dict = lambda *a, **kw: {}
    sys.modules["marker.output"].text_from_rendered = lambda *a, **kw: ("", None, {})
    sys.modules["marker.schema"].BlockTypes = types.SimpleNamespace(
        FigureGroup="FigureGroup",
        TableGroup="TableGroup",
        PictureGroup="PictureGroup",
        Figure="Figure",
        Table="Table",
        Picture="Picture",
        Caption="Caption",
        Footnote="Footnote",
        Text="Text",
    )


def _install_preparse_cache(preparse_cache: Path) -> None:
    """Bypass Marker PDF conversion using cached raw.md and extracted assets."""
    parser_module = importlib.import_module("src.agents.parser")
    parser_class = parser_module.Parser
    load_prompt_fn = parser_module.load_prompt

    def cached_init(self: Any) -> None:
        self.name = "parser"
        self.converter = None
        self.clean_pattern = re.compile(r"<!--[\s\S]*?-->")
        self.enhanced_abt_prompt = load_prompt_fn(
            "config/prompts/narrative_abt_extraction.txt"
        )
        self.visual_classification_prompt = load_prompt_fn(
            "config/prompts/classify_visuals.txt"
        )
        self.title_authors_prompt = load_prompt_fn(
            "config/prompts/extract_title_authors.txt"
        )
        self.section_extraction_prompt = load_prompt_fn(
            "config/prompts/extract_structured_sections.txt"
        )

    def cached_extract_raw_text(
        self: Any, _pdf_path: str, content_dir: Path
    ) -> tuple[str, Any]:
        raw_text = (preparse_cache / "raw.md").read_text(encoding="utf-8")
        text = self.clean_pattern.sub("", raw_text)
        (content_dir / "raw.md").write_text(text, encoding="utf-8")
        print(
            f"[postergen/cache] loaded cached raw.md ({len(text)} chars) from {preparse_cache}",
            flush=True,
        )
        return text, (None, None, {})

    def cached_extract_assets(
        self: Any, _result: Any, _name: str, assets_dir: Path
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        src_assets = preparse_cache / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        for item in sorted(src_assets.iterdir()):
            if item.is_file():
                shutil.copy2(item, assets_dir / item.name)
        figures = json.loads(
            (assets_dir / "figures.json").read_text(encoding="utf-8")
        )
        tables = json.loads(
            (assets_dir / "tables.json").read_text(encoding="utf-8")
        )
        for collection in (figures, tables):
            for entry in collection.values():
                if isinstance(entry, dict) and "path" in entry:
                    entry["path"] = str(assets_dir / Path(entry["path"]).name)
        (assets_dir / "figures.json").write_text(
            json.dumps(figures, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (assets_dir / "tables.json").write_text(
            json.dumps(tables, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"[postergen/cache] loaded {len(figures)} figures and {len(tables)} tables from {src_assets}",
            flush=True,
        )
        return figures, tables

    parser_class.__init__ = cached_init
    parser_class._extract_raw_text = cached_extract_raw_text
    parser_class._extract_assets = cached_extract_assets


def _activate_upstream(upstream: Path, use_preparse_cache: bool = False):
    """Make PosterGen's absolute ``src`` imports win over this repository's package."""

    if use_preparse_cache:
        _install_marker_stubs()
    for name in list(sys.modules):
        if name == "src" or name.startswith("src.") or name == "utils" or name.startswith("utils."):
            del sys.modules[name]
    upstream_text = str(upstream)
    if upstream_text in sys.path:
        sys.path.remove(upstream_text)
    sys.path.insert(0, upstream_text)
    importlib.import_module("utils")
    install_langgraph_shim(upstream)
    return importlib.import_module("src.workflow.pipeline")


def _install_parser_lock(pipeline: Any, lock_path: Path) -> None:
    """Serialize only Marker model loading and PDF extraction on the GPU."""

    parser_module = importlib.import_module("src.agents.parser")
    parser_class = parser_module.Parser
    original_init = parser_class.__init__
    original_extract = parser_class._extract_raw_text

    def release_marker(self: Any) -> None:
        import fcntl
        import gc

        marker_lock = getattr(self, "_codex_marker_lock", None)
        if marker_lock is None:
            return
        self.converter = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
        fcntl.flock(marker_lock.fileno(), fcntl.LOCK_UN)
        marker_lock.close()
        self._codex_marker_lock = None
        print("[postergen/marker] released GPU lock", flush=True)

    def locked_init(self: Any, *args: Any, **kwargs: Any) -> None:
        import fcntl

        lock_path.parent.mkdir(parents=True, exist_ok=True)
        marker_lock = lock_path.open("a+", encoding="utf-8")
        self._codex_marker_lock = marker_lock
        print(f"[postergen/marker] waiting for GPU lock: {lock_path}", flush=True)
        fcntl.flock(marker_lock.fileno(), fcntl.LOCK_EX)
        print("[postergen/marker] acquired GPU lock", flush=True)
        try:
            original_init(self, *args, **kwargs)
        except Exception:
            release_marker(self)
            raise

    def locked_extract(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return original_extract(self, *args, **kwargs)
        finally:
            release_marker(self)

    parser_class.__init__ = locked_init
    parser_class._extract_raw_text = locked_extract


def _install_upstream_compatibility() -> None:
    """Guard two upstream edge cases exposed by scientific-paper content."""

    renderer_class = importlib.import_module("src.agents.renderer").Renderer

    def safe_tokenize_formatting(self: Any, text: str) -> list[dict[str, Any]]:
        """Tokenize supported markup while preserving unmatched markers literally."""

        segments: list[dict[str, Any]] = []

        def append(
            value: str,
            *,
            bold: bool = False,
            italic: bool = False,
            color: str | None = None,
        ) -> None:
            if not value:
                return
            style = {"bold": bold, "italic": italic, "color": color}
            if segments and all(segments[-1][key] == setting for key, setting in style.items()):
                segments[-1]["text"] += value
            else:
                segments.append({"text": value, **style})

        index = 0
        while index < len(text):
            color_match = re.match(r"<color:(#[0-9A-Fa-f]{6})>", text[index:])
            if color_match:
                content_start = index + color_match.end()
                closing_match = re.search(r"</color>", text[content_start:])
                if closing_match:
                    content_end = content_start + closing_match.start()
                    append(text[content_start:content_end], bold=True, color=color_match.group(1))
                    index = content_start + closing_match.end()
                    continue

            bold_match = re.match(r"\*\*(.*?)\*\*", text[index:])
            if bold_match:
                append(bold_match.group(1), bold=True)
                index += bold_match.end()
                continue

            italic_match = re.match(r"\*(.*?)\*", text[index:])
            if italic_match:
                append(italic_match.group(1), italic=True)
                index += italic_match.end()
                continue

            # Upstream advances by zero when an unmatched ``*`` or malformed
            # ``<color:`` starts here. Treat it as literal scientific notation.
            append(text[index])
            index += 1

        return segments

    renderer_class._tokenize_formatting = safe_tokenize_formatting

    layout_class = importlib.import_module("src.agents.layout_agent").LayoutAgent
    original_title_design = layout_class._create_section_title_design

    def safe_title_design(
        self: Any,
        section: dict[str, Any],
        column_x: float,
        start_y: float,
        column_width: float,
        state: Any,
    ) -> list[dict[str, Any]]:
        """Give Balancer-added sections the renderer's existing default style."""

        section_id = section.get("section_id", "")
        design = state.setdefault("section_title_design", {}).setdefault("section_title_design", {})
        applications = design.setdefault("section_applications", [])
        if not any(item.get("section_id") == section_id for item in applications):
            applications.append({"section_id": section_id})
        return original_title_design(self, section, column_x, start_y, column_width, state)

    layout_class._create_section_title_design = safe_title_design


def _prepare_input(workdir: Path, paper: Path) -> Path:
    input_dir = workdir / "input" / paper.parent.name
    input_dir.mkdir(parents=True)
    destination = input_dir / "paper.pdf"
    try:
        destination.symlink_to(paper)
    except OSError:
        shutil.copy2(paper, destination)
    return destination


def _ensure_preview(generated: Path) -> Path:
    previews = sorted(generated.glob("*.png"))
    if previews:
        return previews[0]
    presentations = sorted(generated.glob("*.pptx"))
    if not presentations:
        raise RuntimeError(f"PosterGen produced no PPTX in {generated}")
    pptx = presentations[0]
    libreoffice_profile = generated / ".libreoffice-profile"
    libreoffice_profile.mkdir(exist_ok=True)
    lo_bin = shutil.which("libreoffice") or shutil.which("soffice") or "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    completed = subprocess.run(
        [
            lo_bin,
            f"-env:UserInstallation={libreoffice_profile.resolve().as_uri()}",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(generated),
            str(pptx),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    pdf = generated / f"{pptx.stem}.pdf"
    if completed.returncode != 0 or not pdf.is_file():
        raise RuntimeError(f"LibreOffice preview conversion failed: {completed.stderr[-1000:]}")
    completed = subprocess.run(
        ["pdftoppm", "-png", "-singlefile", "-r", "150", str(pdf), str(generated / pptx.stem)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    preview = generated / f"{pptx.stem}.png"
    if completed.returncode != 0 or not preview.is_file():
        raise RuntimeError(f"pdftoppm preview conversion failed: {completed.stderr[-1000:]}")
    return preview


def _git_revision(upstream: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _publish(
    args: argparse.Namespace,
    generated: Path,
    trace_dir: Path | None,
    output: Path,
    paper: Path,
    upstream: Path,
    started_at: str,
    duration_seconds: float,
    codex_usage: dict[str, Any],
) -> None:
    if trace_dir is not None and trace_dir.is_dir():
        shutil.move(str(trace_dir), str(generated / "codex_traces"))
    if output.exists():
        if any(output.iterdir()):
            if not args.force:
                raise RuntimeError(f"Refusing to replace non-empty output: {output}")
            shutil.rmtree(output)
        else:
            output.rmdir()
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(generated), str(output))
    manifest = {
        "baseline": "PosterGen",
        "model_backend": "codex exec",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration_seconds, 2),
        "codex_usage": codex_usage,
        "codex_model": args.codex_model,
        "reasoning_effort": args.reasoning_effort,
        "paper": str(paper),
        "upstream_root": str(upstream),
        "upstream_revision": _git_revision(upstream),
        "poster_width": args.poster_width,
        "poster_height": args.poster_height,
        "logo": str(args.logo.expanduser().resolve()) if args.logo else None,
        "affiliation_logo": str(args.aff_logo.expanduser().resolve()) if args.aff_logo else None,
        "traces_preserved": not args.no_traces,
        "global_usage_ledger": None if args.no_global_ledger else str(args.usage_ledger.expanduser()),
    }
    (output / "adapter_run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paper, upstream, output = _validate(args)
    original_cwd = Path.cwd()

    if args.check:
        os.chdir(upstream)
        try:
            _activate_upstream(upstream, use_preparse_cache=args.preparse_cache is not None)
        except ModuleNotFoundError as exc:
            raise SystemExit(
                f"Missing PosterGen dependency: {exc.name}. Install "
                "src/baseline/postergen/requirements.txt in the selected environment."
            ) from exc
        finally:
            os.chdir(original_cwd)
        print(f"PosterGen: {upstream}")
        print(f"Paper: {paper}")
        print(f"Output: {output}")
        print(f"Codex: {shutil.which(args.codex_bin)}")
        print("Adapter paths and upstream imports passed.")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f".postergen-{paper.parent.name[:40]}-", dir=output.parent))
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    succeeded = False
    try:
        input_paper = _prepare_input(workdir, paper)
        trace_dir = None if args.no_traces else workdir / "codex_traces"
        ledger_path = None if args.no_global_ledger else args.usage_ledger.expanduser()
        configure_codex(
            CodexRuntime(
                executable=args.codex_bin,
                model=args.codex_model,
                reasoning_effort=args.reasoning_effort,
                timeout_s=args.timeout,
                cwd=workdir,
                trace_dir=trace_dir,
                application="postergen",
                log_prefix="postergen/codex",
                ledger_path=ledger_path,
                result_dir=output,
            )
        )
        os.chdir(workdir)
        pipeline = _activate_upstream(
            upstream, use_preparse_cache=args.preparse_cache is not None
        )
        _install_upstream_compatibility()
        if args.preparse_cache is not None:
            _install_preparse_cache(args.preparse_cache)
        elif not args.no_parser_lock:
            _install_parser_lock(pipeline, args.parser_lock.expanduser().resolve())
        state = pipeline.create_state(
            str(input_paper),
            args.codex_model,
            args.codex_model,
            args.poster_width,
            args.poster_height,
            args.url,
            str(args.logo.expanduser().resolve()) if args.logo else "",
            str(args.aff_logo.expanduser().resolve()) if args.aff_logo else "",
        )
        for model_key in ("text_model", "vision_model"):
            state[model_key].model_name = args.codex_model
            state[model_key].provider = "codex_exec"
        state["timing_metrics"].pipeline_start = time.time()
        final_state = pipeline.create_workflow_graph().compile().invoke(state)
        final_state["timing_metrics"].pipeline_end = time.time()
        if final_state.get("errors"):
            raise RuntimeError(f"PosterGen pipeline errors: {final_state['errors']}")
        required = ("story_board", "design_layout", "color_scheme", "styled_layout")
        missing = [name for name in required if not final_state.get(name)]
        if missing:
            raise RuntimeError(f"PosterGen missing outputs: {missing}")
        pipeline.save_timing_log(final_state)
        generated = workdir / "output" / paper.parent.name
        if not generated.is_dir():
            raise RuntimeError(f"PosterGen produced no output directory: {generated}")
        _ensure_preview(generated)
        duration_seconds = time.monotonic() - started
        codex_usage = get_codex_run_stats()
        _publish(
            args,
            generated,
            trace_dir,
            output,
            paper,
            upstream,
            started_at,
            duration_seconds,
            codex_usage,
        )
        succeeded = True
        print(
            f"PosterGen metrics: {duration_seconds:.1f}s, calls={codex_usage['call_count']}, "
            f"api-equivalent=${codex_usage['estimated_api_cost_usd']:.4f}"
        )
        print(f"PosterGen output: {output}")
        return 0
    finally:
        os.chdir(original_cwd)
        if succeeded and not args.keep_workdir:
            shutil.rmtree(workdir)
        else:
            print(f"PosterGen workdir retained at: {workdir}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
