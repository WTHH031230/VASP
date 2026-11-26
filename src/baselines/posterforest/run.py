"""Run PosterForest with Codex transport and audited order-only compatibility fixes."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import importlib.metadata
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

from src.baselines.common.codex import CodexRuntime, configure_codex, get_codex_run_stats
from src.baselines.common.manifest import write_manifest
from src.baselines.posterforest.camel_backend import install, link_runtime_files


REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_ROOT = REPO_ROOT / "third_party" / "PosterForest"
UPSTREAM_MAIN = UPSTREAM_ROOT / "PosterForest" / "main.py"
UPSTREAM_STEP_04 = UPSTREAM_ROOT / "PosterForest" / "step_04_layout_hierarchy.py"
UPSTREAM_STEP_06 = UPSTREAM_ROOT / "PosterForest" / "step_06_modification_planning.py"


def _install_transport_timeout(timeout_s: int) -> None:
    """Match upstream's agent wait deadline to the Codex transport deadline."""

    from utils import wei_utils

    wei_utils._AGENT_STEP_TIMEOUT_TEXT = timeout_s
    wei_utils._AGENT_STEP_TIMEOUT_VISION = timeout_s


def _install_step_04_table_ratio_fix() -> None:
    """Initialize the shared debug ratio on the upstream table branch.

    The table branch computes ``table_size_ratio`` and all layout values from
    it, but the common diagnostic print below the branch reads ``size_ratio``.
    Aliasing the already-computed value prevents a logging-only
    use-before-assignment without changing any geometry or threshold.
    """

    module_name = "PosterForest.step_04_layout_hierarchy"
    importlib.import_module(module_name)
    source = UPSTREAM_STEP_04.read_text(encoding="utf-8")
    marker = "                                leaf['table_size_ratio'] = table_size_ratio\n"
    replacement = marker + "                                size_ratio = table_size_ratio\n"
    if source.count(marker) != 1:
        raise RuntimeError("PosterForest step 04 compatibility marker changed upstream")
    source = source.replace(marker, replacement, 1)
    spec = importlib.util.spec_from_file_location(module_name, UPSTREAM_STEP_04)
    if spec is None:
        raise RuntimeError(f"Cannot create module spec for {UPSTREAM_STEP_04}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(UPSTREAM_STEP_04), "exec"), module.__dict__)
        setattr(sys.modules["PosterForest"], "step_04_layout_hierarchy", module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise


def _install_step_06_initialization_fix() -> None:
    """Load upstream step 06 with its natural-size values initialized before use.

    Commit dd4b060 computes the four ``_nat_fig_*`` values below the figure
    recovery block that first reads them.  Moving the same expressions above
    that block restores the evident intended execution order without changing
    any layout formula or decision threshold.  The patched module exists only
    in this adapter process; the third-party checkout remains byte-for-byte
    unchanged.
    """

    module_name = "PosterForest.step_06_modification_planning"
    # Initializing the parent package first avoids its eager ``__init__``
    # imports observing a half-constructed replacement module.
    importlib.import_module(module_name)
    source = UPSTREAM_STEP_06.read_text(encoding="utf-8")
    marker = "                # Figure recovery: when raw_dir is still 'none'"
    initialization = """                # Adapter compatibility: upstream dd4b060 reads these values
                # in the following recovery block before assigning them later.
                _nat_fig_h_L = max((float(_f.get('height', 0)) for _f in test_figure_arrangement
                                    if str(_f.get('panel_id', '')) in _left_ids), default=0.0)
                _nat_fig_h_R = max((float(_f.get('height', 0)) for _f in test_figure_arrangement
                                    if str(_f.get('panel_id', '')) in _right_ids), default=0.0)
                _nat_fig_w_L = max((float(_f.get('width', 0)) for _f in test_figure_arrangement
                                    if str(_f.get('panel_id', '')) in _left_ids), default=0.0)
                _nat_fig_w_R = max((float(_f.get('width', 0)) for _f in test_figure_arrangement
                                    if str(_f.get('panel_id', '')) in _right_ids), default=0.0)

"""
    if source.count(marker) != 1:
        raise RuntimeError("PosterForest step 06 compatibility marker changed upstream")
    source = source.replace(marker, initialization + marker, 1)
    spec = importlib.util.spec_from_file_location(module_name, UPSTREAM_STEP_06)
    if spec is None:
        raise RuntimeError(f"Cannot create module spec for {UPSTREAM_STEP_06}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(UPSTREAM_STEP_06), "exec"), module.__dict__)
        setattr(
            sys.modules["PosterForest"],
            "step_06_modification_planning",
            module,
        )
    except Exception:
        sys.modules.pop(module_name, None)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument(
        "--upstream-model-alias",
        default="4o",
        help="PosterForest control-flow/config alias; calls still use --model",
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument(
        "--normalize-pdf-with-ghostscript",
        action="store_true",
        help="Rewrite a structurally incompatible PDF before upstream Docling parsing",
    )
    return parser.parse_args()


def _pdf_page_count(path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"pdfinfo did not report a page count for {path}")


def _normalize_pdf_with_ghostscript(paper: Path, work_dir: Path) -> Path:
    """Canonicalize malformed internal PDF geometry without rasterizing pages."""

    normalized = work_dir / "_adapter_input" / paper.parent.name / "paper.pdf"
    normalized.parent.mkdir(parents=True, exist_ok=True)
    source_meta = paper.parent / "meta.json"
    if source_meta.is_file():
        destination_meta = normalized.parent / "meta.json"
        if not destination_meta.exists():
            shutil.copy2(source_meta, destination_meta)
    subprocess.run(
        [
            "gs",
            "-q",
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.5",
            "-dPDFSETTINGS=/prepress",
            "-dDetectDuplicateImages=true",
            f"-sOutputFile={normalized}",
            str(paper),
        ],
        timeout=1800,
        check=True,
    )
    if not normalized.is_file() or normalized.stat().st_size == 0:
        raise RuntimeError("Ghostscript produced no normalized PDF")
    source_pages = _pdf_page_count(paper)
    normalized_pages = _pdf_page_count(normalized)
    if normalized_pages != source_pages:
        raise RuntimeError(
            f"PDF normalization changed page count: {source_pages} -> {normalized_pages}"
        )
    return normalized


def _latest_output(work_dir: Path) -> Path | None:
    candidates = [path for path in (work_dir / "outputs").glob("*") if path.is_dir()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _normalize_outputs(
    upstream_output: Path | None,
    output_dir: Path,
) -> tuple[Path | None, Path | None]:
    if upstream_output is None:
        return None, None
    jpg = upstream_output / "08_finalize_output" / "poster_final.jpg"
    pptx = upstream_output / "08_finalize_output" / "poster_final.pptx"
    png_path: Path | None = None
    pptx_path: Path | None = None
    if jpg.is_file():
        png_path = output_dir / "poster.png"
        with Image.open(jpg) as image:
            image.convert("RGB").save(png_path, "PNG")
    if pptx.is_file():
        pptx_path = output_dir / "poster.pptx"
        shutil.copy2(pptx, pptx_path)
    return png_path, pptx_path


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _format_error(exc: BaseException) -> str:
    """Expose exceptions hidden by upstream Tenacity wrappers."""

    chain: list[BaseException] = [exc]
    last_attempt = getattr(exc, "last_attempt", None)
    if last_attempt is not None:
        try:
            nested = last_attempt.exception()
        except Exception:
            nested = None
        if isinstance(nested, BaseException) and nested is not exc:
            chain.append(nested)
    current = chain[-1]
    while len(chain) < 5:
        nested = current.__cause__ or current.__context__
        if nested is None or nested in chain:
            break
        chain.append(nested)
        current = nested
    return " <- ".join(f"{type(item).__name__}: {item}" for item in chain)


def run(args: argparse.Namespace) -> int:
    paper = args.paper.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not paper.is_file():
        raise SystemExit(f"Paper not found: {paper}")
    if (args.width is None) != (args.height is None):
        raise SystemExit("--width and --height must be supplied together")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to replace non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_upstream"
    trace_dir = output_dir / "codex_traces"
    link_runtime_files(work_dir, UPSTREAM_ROOT)
    configure_codex(
        CodexRuntime(
            executable=args.codex_bin,
            model=args.model,
            reasoning_effort=args.effort,
            timeout_s=args.timeout,
            cwd=UPSTREAM_ROOT,
            trace_dir=trace_dir,
            application="posterforest-strict",
            log_prefix="posterforest-strict/codex",
            result_dir=output_dir,
        )
    )
    install()
    if str(UPSTREAM_ROOT) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_ROOT))
    _install_transport_timeout(args.timeout)
    _install_step_04_table_ratio_fix()
    _install_step_06_initialization_fix()

    status = "failed"
    error: str | None = None
    started = time.monotonic()
    previous_cwd = Path.cwd()
    previous_argv = sys.argv[:]
    try:
        pipeline_paper = (
            _normalize_pdf_with_ghostscript(paper, work_dir)
            if args.normalize_pdf_with_ghostscript
            else paper
        )
        os.chdir(work_dir)
        argv = [
            str(UPSTREAM_MAIN),
            "--paper_path",
            str(pipeline_paper),
            "--model_name_t",
            args.upstream_model_alias,
            "--model_name_v",
            args.upstream_model_alias,
        ]
        if args.width is not None:
            argv.extend(
                [
                    "--poster_width_inches",
                    str(args.width),
                    "--poster_height_inches",
                    str(args.height),
                ]
            )
        sys.argv = argv
        # Docling and all eight official PosterForest steps run from upstream.
        # The loader above only corrects an upstream use-before-assignment bug.
        runpy.run_path(str(UPSTREAM_MAIN), run_name="__main__")
        status = "completed"
    except SystemExit as exc:
        if exc.code in (None, 0):
            status = "completed"
        else:
            error = f"PosterForest exited with status {exc.code}"
    except Exception as exc:
        error = _format_error(exc)
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)

    upstream_output = _latest_output(work_dir)
    png_path, pptx_path = _normalize_outputs(upstream_output, output_dir)
    upstream_error = error
    if png_path is not None:
        status = "completed"
        error = None
    elif status == "completed":
        status = "failed"
        error = error or "PosterForest completed without poster_final.jpg"
    wall_seconds = round(time.monotonic() - started, 3)
    usage = get_codex_run_stats()
    requested_size = [args.width, args.height] if args.width is not None else None
    write_manifest(
        output_dir / "adapter_run.json",
        {
            "method": "PosterForest",
            "upstream": "kaist-cvml/poster-forest",
            "upstream_commit": "dd4b060",
            "strict_upstream": True,
            "only_adaptation": (
                "CAMEL model -> codex exec --ephemeral; in-memory initialization of "
                "upstream dd4b060 step-04 table debug ratio and relocation of "
                "step-06 _nat_fig_* initialization before first use"
            ),
            "upstream_compatibility_fixes": [
                {
                    "file": "utils/wei_utils.py",
                    "kind": "transport_timeout_mapping",
                    "upstream_text_timeout_seconds": 120,
                    "upstream_vision_timeout_seconds": 180,
                    "codex_transport_timeout_seconds": args.timeout,
                    "semantic_change": False,
                },
                {
                    "file": "PosterForest/step_04_layout_hierarchy.py",
                    "kind": "logging_use_before_assignment",
                    "variable": "size_ratio",
                    "source_variable": "table_size_ratio",
                    "semantic_change": False,
                },
                {
                    "file": "PosterForest/step_06_modification_planning.py",
                    "kind": "use_before_assignment",
                    "variables": [
                        "_nat_fig_h_L",
                        "_nat_fig_h_R",
                        "_nat_fig_w_L",
                        "_nat_fig_w_R",
                    ],
                    "semantic_change": False,
                }
            ],
            "status": status,
            "error": error,
            "upstream_error": upstream_error,
            "result_acceptance_policy": "first_available_output_no_quality_filter",
            "paper": str(paper),
            "input_policy": (
                "ghostscript_structural_normalization_then_upstream_docling_and_caption_asset_extraction"
                if args.normalize_pdf_with_ghostscript
                else "upstream_docling_and_caption_asset_extraction"
            ),
            "pdf_structural_normalization": args.normalize_pdf_with_ghostscript,
            "pdf_normalization_semantic_change": False,
            "parser_backend": "docling",
            "docling_version": _package_version("docling"),
            "docling_core_version": _package_version("docling-core"),
            "docling_parse_version": _package_version("docling-parse"),
            "docling_ibm_models_version": _package_version("docling-ibm-models"),
            "transformers_version": _package_version("transformers"),
            "parser_environment_resolution": (
                "docling 2.49.0 is the final release before Heron/RT-DETR-v2; "
                "it is compatible with upstream transformers 4.48.0 and the "
                "README-pinned docling-parse 4.5.0"
            ),
            "model_backend": "codex exec --ephemeral",
            "model": args.model,
            "reasoning_effort": args.effort,
            "upstream_model_alias": args.upstream_model_alias,
            "poster_size_request_inches": requested_size,
            "poster_size_policy": (
                "explicit_official_cli"
                if requested_size
                else "upstream_meta_json_then_48x36_fallback"
            ),
            "total_wall_seconds": wall_seconds,
            "api_equivalent_cost_usd": usage["estimated_api_cost_usd"],
            "codex_usage": usage,
            "upstream_output": str(upstream_output) if upstream_output else None,
            "poster_png": str(png_path) if png_path else None,
            "poster_pptx": str(pptx_path) if pptx_path else None,
        },
    )
    if error:
        print(f"PosterForest failed: {error}", file=sys.stderr)
    else:
        print(
            f"PosterForest completed: {wall_seconds:.1f}s, "
            f"calls={usage['call_count']}, "
            f"api-equivalent=${usage['estimated_api_cost_usd']:.4f}"
        )
    return 0 if status == "completed" else 1


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
