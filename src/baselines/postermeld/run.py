"""Run the official PosterMELD benchmark path with Codex as the model transport.

The adapter calls PosterMELD's own CLI entry point.  Its MinerU-to-Marker parser,
workflow graph, layout selection, reviews, repair loops, quality gate, and
benchmark configuration are left unchanged.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import importlib.metadata
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from src.baselines.common.codex import (
    CodexRuntime,
    configure_codex,
    get_codex_run_stats,
    invoke_codex,
)
from src.baselines.common.manifest import write_manifest
from src.baselines.postergen.codex_backend import install_langgraph_shim


REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_ROOT = REPO_ROOT / "third_party" / "PosterMELD" / "poster_generation"
LOCAL_AFFILIATION_LOGO_NAMES = (
    "affiliation_logo.png",
    "affiliation-logo.png",
    "aff_logo.png",
    "aff.png",
    "affiliation_logo.jpg",
    "affiliation_logo.jpeg",
    "affiliation_logo.webp",
)
DEFAULT_PRECOMPUTED_MINERU_ROOT = Path(
    os.environ.get("MINERU_PRELABEL_DIR", "./mineru_prelabel")
)
REFERENCE_MINERU_SUCCESS_MEAN_SECONDS = 46.72
REFERENCE_MINERU_BATCH_TOTAL_SECONDS = 4509.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--conference", default="")
    parser.add_argument("--poster-width", type=float)
    parser.add_argument("--poster-height", type=float)
    parser.add_argument(
        "--mineru-base-url",
        default=os.getenv("MINERU_BASE_URL", "http://127.0.0.1:8011"),
        help="MinerU cloud-v4 endpoint; defaults to the local compatibility bridge",
    )
    parser.add_argument(
        "--mineru-api-key",
        default=os.getenv("MINERU_API_KEY", "local"),
        help="MinerU credential; the localhost bridge accepts the non-secret value 'local'",
    )
    parser.add_argument(
        "--mineru-precomputed-root",
        type=Path,
        help="Numbered MinerU output bundles to load instead of calling an API",
    )
    parser.add_argument("--keep-workdir", action="store_true")
    return parser


class CodexVLMResponse:
    """Minimal requests.Response surface used by the upstream VLM reviewer."""

    headers = {"content-type": "application/json"}

    def __init__(self, answer: str, usage: dict[str, int]):
        self.answer = answer
        self.usage = usage

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": self.answer}}],
            "usage": self.usage,
        }


def _codex_vlm_request(
    self: Any,
    base_url: str,
    headers: dict[str, str],
    model: str,
    prompt: str,
    image_data: str,
    *,
    transport: str | None = None,
) -> CodexVLMResponse:
    del self, base_url, headers, model, transport
    encoded = image_data.split(",", 1)[-1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as source:
        image = source.convert("RGB").copy()
    try:
        answer, usage, _ = invoke_codex(prompt, [image])
    finally:
        image.close()
    return CodexVLMResponse(answer, usage)


def _activate_upstream():
    """Import pristine upstream code, replacing model transports only."""

    for name in list(sys.modules):
        if (
            name == "src"
            or name.startswith("src.")
            or name == "utils"
            or name.startswith("utils.")
        ):
            del sys.modules[name]
    upstream_text = str(UPSTREAM_ROOT)
    if upstream_text in sys.path:
        sys.path.remove(upstream_text)
    sys.path.insert(0, upstream_text)
    importlib.import_module("utils")
    install_langgraph_shim(UPSTREAM_ROOT, application_name="PosterMELD")
    reviewer = importlib.import_module("src.agents.vlm_layout_reviewer")
    reviewer.VLMLayoutReviewer._post_vlm_request = _codex_vlm_request
    return importlib.import_module("src.workflow.pipeline")


def _reference_bundle_for_paper(root: Path, paper: Path) -> Path:
    dataset_papers = sorted(paper.parent.parent.glob("*/paper.pdf"))
    resolved = paper.resolve()
    try:
        paper_index = next(
            index
            for index, candidate in enumerate(dataset_papers, 1)
            if candidate.resolve() == resolved
        )
    except StopIteration as exc:
        raise RuntimeError(f"Could not determine dataset index for {paper}") from exc
    candidates = sorted(root.glob(f"{paper_index:03d}_*"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one precomputed MinerU bundle for index {paper_index:03d}, "
            f"found {len(candidates)} under {root}"
        )
    bundle = candidates[0]
    if not (bundle / "mineru_raw").is_dir():
        raise RuntimeError(f"Precomputed MinerU bundle has no mineru_raw: {bundle}")
    return bundle


def _install_precomputed_mineru(
    bundle: Path,
    phase_clock: dict[str, float],
) -> None:
    """Serve one existing MinerU bundle through the upstream client object."""

    mineru_api = importlib.import_module("src.tools.mineru_api")
    source_raw = bundle / "mineru_raw"
    markdown_candidates = sorted(source_raw.rglob("paper.md"))
    content_candidates = sorted(
        path
        for path in source_raw.rglob("*_content_list.json")
        if not path.name.endswith("_v2.json")
    )
    if len(markdown_candidates) != 1 or len(content_candidates) != 1:
        raise RuntimeError(
            f"Ambiguous precomputed MinerU bundle {bundle}: "
            f"markdown={len(markdown_candidates)} content_list={len(content_candidates)}"
        )
    source_markdown = markdown_candidates[0]
    source_content_list = content_candidates[0]

    def parse_precomputed(self: Any, pdf_path: str | Path, content_dir: str | Path):
        del self
        pdf_path = Path(pdf_path)
        content_dir = Path(content_dir)
        extract_dir = content_dir / "mineru_raw"
        extract_dir.mkdir(parents=True, exist_ok=True)
        source_paper_dir = source_raw / "paper"
        local_paper_dir = extract_dir / "paper"
        if not local_paper_dir.exists():
            local_paper_dir.symlink_to(source_paper_dir, target_is_directory=True)

        local_markdown = extract_dir / source_markdown.relative_to(source_raw)
        local_content_list = extract_dir / source_content_list.relative_to(source_raw)
        raw_text = local_markdown.read_text(encoding="utf-8", errors="replace")
        loaded = json.loads(local_content_list.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise RuntimeError(f"MinerU content list is not a list: {local_content_list}")
        content_items = [item for item in loaded if isinstance(item, dict)]
        content_dir.mkdir(parents=True, exist_ok=True)
        (content_dir / "raw.md").write_text(raw_text, encoding="utf-8")
        report = {
            "backend": "mineru",
            "model_version": "pipeline",
            "base_url": None,
            "pdf_name": pdf_path.name,
            "fallback_used": False,
            "state": "completed_precomputed",
            "precomputed": True,
            "precomputed_bundle": str(bundle),
            "markdown_path": str(local_markdown),
            "content_list_path": str(local_content_list),
            "content_item_count": len(content_items),
            "mineru_parse_time_seconds": REFERENCE_MINERU_SUCCESS_MEAN_SECONDS,
            "mineru_parse_time_kind": "reference_success_mean_estimate",
            "mineru_batch_total_seconds": REFERENCE_MINERU_BATCH_TOTAL_SECONDS,
        }
        phase_clock["post_mineru_started"] = time.monotonic()
        return mineru_api.MinerUExtraction(
            raw_text=raw_text,
            extract_dir=extract_dir,
            zip_path=extract_dir / "precomputed_mineru_result.zip",
            content_list_path=local_content_list,
            content_items=content_items,
            pdf_path=pdf_path,
            report=report,
        )

    mineru_api.MinerUClient.parse_pdf = parse_precomputed


def _install_rejected_draft_export(
    pipeline: Any,
    report: dict[str, Any],
) -> None:
    """Render one faithful artifact for layouts rejected before rendering.

    PosterMELD normally terminates immediately when the micro-layout refiner
    rejects a state.  For benchmark observability, keep that exact rejected
    state, resolve its existing visual assets, render it once, and stop.  No
    repair, resampling, review, or quality acceptance is introduced.
    """

    original_save_timing_log = pipeline.save_timing_log
    def save_timing_log(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("draft_status") == "rejected":
            report["triggered"] = True
            report["rejection_reason"] = state.get("draft_rejection_reason")
            report["_rejected_state"] = state
        return original_save_timing_log(state)
    pipeline.save_timing_log = save_timing_log


def _render_captured_rejected_draft(report: dict[str, Any]) -> None:
    state = report.pop("_rejected_state", None)
    if not isinstance(state, dict):
        return
    visual_module = importlib.import_module("src.agents.visual_asset_agent")
    renderer_module = importlib.import_module("src.agents.renderer")
    visual_agent = visual_module.VisualAssetAgent()
    state = visual_agent(state)
    if state.get("visual_reflow_required") and not state.get("resolved_visual_assets"):
        # Keep the rejected geometry. Resolve the existing plan directly rather
        # than following PosterMELD's normal layout-repair/reflow edge.
        report["visual_reflow_suppressed"] = True
        state["resolved_visual_assets"] = visual_agent._execute_plan(
            state.get("visual_plan") or [],
            state,
        )
        state["visual_reflow_required"] = False
        visual_agent._save_outputs(state)
    state = renderer_module.Renderer()(state)
    render_errors = [
        item
        for item in state.get("errors", [])
        if str(item).startswith(("visual_asset_agent:", "renderer:"))
    ]
    if render_errors:
        raise RuntimeError("; ".join(str(item) for item in render_errors))
    report["rendered"] = bool(state.get("pptx_output_path"))


def _prepare_input(workdir: Path, paper: Path) -> Path:
    paper_dir = workdir / "input" / paper.parent.name
    paper_dir.mkdir(parents=True)
    destination = paper_dir / "paper.pdf"
    try:
        destination.symlink_to(paper)
    except OSError:
        shutil.copy2(paper, destination)
    return destination


def _find_local_affiliation_logo(paper: Path) -> Path | None:
    paper_dir = paper.parent
    for filename in LOCAL_AFFILIATION_LOGO_NAMES:
        candidate = paper_dir / filename
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    logo_dir = paper_dir / "affiliation_logos"
    if logo_dir.is_dir():
        candidates = sorted(
            path
            for path in logo_dir.iterdir()
            if path.is_file()
            and path.stat().st_size > 0
            and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if len(candidates) == 1:
            return candidates[0]
    return None


def _has_no_affiliation_logo_marker(paper: Path) -> bool:
    return any(
        (paper.parent / name).is_file()
        for name in ("no_affiliation_logo", ".no_affiliation_logo")
    )


def _official_benchmark_argv(
    input_paper: Path,
    model: str,
    conference: str,
    poster_width: float | None = None,
    poster_height: float | None = None,
) -> tuple[list[str], str]:
    if (poster_width is None) != (poster_height is None):
        raise ValueError("poster width and height must be supplied together")
    local_logo = _find_local_affiliation_logo(input_paper.resolve())
    logos_disabled = bool(local_logo) or _has_no_affiliation_logo_marker(input_paper.resolve())
    argv = [
        str(input_paper),
        "--text_model",
        model,
        "--vision_model",
        model,
        "--vlm-model",
        model,
        "--layout-template",
        "auto",
    ]
    if conference:
        argv.extend(["--conference", conference])
    if poster_width is not None and poster_height is not None:
        argv.extend(
            [
                "--poster_width",
                str(poster_width),
                "--poster_height",
                str(poster_height),
            ]
        )
    argv.extend(
        [
            "--disable-affiliation-logos" if logos_disabled else "--enable-affiliation-logos",
            "--affiliation-logo-mode",
            "single",
            "--poster-style",
            "navy_serif",
            "--visual-density",
            "balanced",
            "--section-title-numbering",
            "off",
            "--header-seed",
            "20260710",
            "--enable-vlm-layout-review",
            "--enable-visual-legibility-review",
            "--enable-block-vlm-review",
            "--enable-adaptive-column-width",
            "--disable-generated-background",
            "--disable-generated-teaser",
        ]
    )
    if local_logo:
        argv.extend(["--aff-logo", str(local_logo)])
    policy = "local" if local_logo else "disabled" if logos_disabled else "automatic"
    return argv, policy


def _valid_final_pngs(generated: Path) -> list[Path]:
    valid: list[Path] = []
    for path in sorted(generated.glob("*.png")):
        if "draft" in path.stem.lower() or path.stat().st_size <= 0:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            valid.append(path)
        except Exception:
            continue
    return valid


def _valid_draft_pngs(generated: Path) -> list[Path]:
    valid: list[Path] = []
    for path in sorted(generated.glob("*.png")):
        if "draft" not in path.stem.lower() or path.stat().st_size <= 0:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            valid.append(path)
        except Exception:
            continue
    return valid


def _render_pptx_preview(generated: Path, *, draft: bool = False) -> Path:
    presentations = [
        path
        for path in sorted(generated.glob("*.pptx"))
        if ("draft" in path.stem.lower()) is draft
    ]
    if not presentations:
        stage = "draft" if draft else "final"
        raise RuntimeError(f"PosterMELD produced no {stage} PNG or PPTX in {generated}")
    pptx = presentations[0]
    completed = subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(generated),
            str(pptx),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    pdf = generated / f"{pptx.stem}.pdf"
    if completed.returncode != 0 or not pdf.is_file():
        raise RuntimeError(f"LibreOffice preview failed: {completed.stderr[-1000:]}")
    completed = subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-singlefile",
            "-r",
            "150",
            str(pdf),
            str(generated / pptx.stem),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    preview = generated / f"{pptx.stem}.png"
    if completed.returncode != 0 or not preview.is_file():
        raise RuntimeError(f"pdftoppm preview failed: {completed.stderr[-1000:]}")
    return preview


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _endpoint_metadata(base_url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=10) as response:
            value = json.load(response)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _direct_url_metadata(distribution: str) -> dict[str, Any]:
    try:
        dist = importlib.metadata.distribution(distribution)
        text = dist.read_text("direct_url.json")
        value = json.loads(text) if text else {}
        return value if isinstance(value, dict) else {}
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        return {}


def run(args: argparse.Namespace) -> int:
    paper = args.paper.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not paper.is_file():
        raise SystemExit(f"Paper not found: {paper}")
    precomputed_root = args.mineru_precomputed_root.expanduser().resolve() if args.mineru_precomputed_root else None
    precomputed_bundle = _reference_bundle_for_paper(precomputed_root, paper) if precomputed_root else None
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to replace non-empty output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(
        tempfile.mkdtemp(prefix=f".postermeld-strict-{paper.parent.name[:32]}-", dir=output.parent)
    )
    input_paper = _prepare_input(workdir, paper)
    trace_dir = workdir / "codex_traces"
    configure_codex(
        CodexRuntime(
            executable=args.codex_bin,
            model=args.model,
            reasoning_effort=args.effort,
            timeout_s=args.timeout,
            cwd=UPSTREAM_ROOT,
            trace_dir=trace_dir,
            application="postermeld-strict",
            log_prefix="postermeld-strict/codex",
            result_dir=workdir,
        )
    )

    os.environ["PAPER2POSTER_OUTPUT_ROOT"] = str(workdir / "output")
    os.environ["PAPER2POSTER_GENERATED_BACKGROUND"] = "0"
    os.environ["PAPER2POSTER_GENERATED_TEASER"] = "0"
    os.environ["PAPER2POSTER_ALLOW_GENERATIVE_FALLBACK"] = "0"
    os.environ["MINERU_BASE_URL"] = args.mineru_base_url
    os.environ["MINERU_API_KEY"] = args.mineru_api_key
    os.environ["VLM_BASE_URL"] = "codex://local"
    os.environ["VLM_API_KEY"] = "codex-exec-local"
    os.environ["VLM_MODEL"] = args.model
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    status = "failed"
    error: str | None = None
    upstream_exit_code: int | None = None
    affiliation_logo_policy = "unknown"
    generated = workdir / "output" / input_paper.parent.name
    artifact_stage: str | None = None
    previous_cwd = Path.cwd()
    previous_argv = sys.argv[:]
    started = time.monotonic()
    phase_clock: dict[str, float] = {}
    rejected_draft_report: dict[str, Any] = {"triggered": False}
    try:
        pipeline = _activate_upstream()
        if precomputed_bundle is not None:
            _install_precomputed_mineru(precomputed_bundle, phase_clock)
        _install_rejected_draft_export(pipeline, rejected_draft_report)
        cli, affiliation_logo_policy = _official_benchmark_argv(
            input_paper,
            args.model,
            args.conference,
            args.poster_width,
            args.poster_height,
        )
        os.chdir(workdir)
        sys.argv = [str(UPSTREAM_ROOT / "src" / "workflow" / "pipeline.py"), *cli]
        result = pipeline.main()
        if rejected_draft_report.get("triggered"):
            _render_captured_rejected_draft(rejected_draft_report)
        upstream_exit_code = int(result or 0)
        final_pngs = _valid_final_pngs(generated)
        if not final_pngs:
            try:
                _render_pptx_preview(generated)
            except RuntimeError:
                pass
            final_pngs = _valid_final_pngs(generated)
        selected_png = final_pngs[0] if final_pngs else None
        if selected_png is not None:
            artifact_stage = "final"
        else:
            draft_pngs = _valid_draft_pngs(generated)
            if not draft_pngs:
                try:
                    _render_pptx_preview(generated, draft=True)
                except RuntimeError:
                    pass
                draft_pngs = _valid_draft_pngs(generated)
            if draft_pngs:
                selected_png = draft_pngs[0]
                artifact_stage = "draft"
        if selected_png is None:
            raise RuntimeError("PosterMELD produced no valid poster PNG or PPTX")
        # Preserve the first available poster artifact. Quality-gate status is
        # recorded for analysis but never used to resample a completed poster.
        output.mkdir(parents=True, exist_ok=True)
        shutil.copytree(generated, output, dirs_exist_ok=True)
        shutil.copy2(output / selected_png.name, output / "poster.png")
        presentations = [
            path
            for path in sorted(output.glob("*.pptx"))
            if ("draft" in path.stem.lower()) is (artifact_stage == "draft")
        ]
        if presentations:
            shutil.copy2(presentations[0], output / "poster.pptx")
        if rejected_draft_report.get("triggered"):
            artifact_stage = "rejected_draft"
        if trace_dir.is_dir():
            shutil.copytree(trace_dir, output / "codex_traces", dirs_exist_ok=True)
        status = "completed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        output.mkdir(parents=True, exist_ok=True)
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)

    finished = time.monotonic()
    wall_seconds = round(finished - started, 3)
    post_mineru_generation_seconds = (
        round(finished - phase_clock["post_mineru_started"], 3)
        if "post_mineru_started" in phase_clock
        else None
    )
    usage = get_codex_run_stats()
    quality_gate = _read_json(output / "content" / "final_quality_gate.json")
    parser_report = _read_json(output / "content" / "mineru_report.json")
    mineru_endpoint_metadata = {} if precomputed_bundle is not None else _endpoint_metadata(args.mineru_base_url)
    parser_backend = str(parser_report.get("backend") or "marker")
    if parser_report.get("fallback_used"):
        parser_backend = str(parser_report.get("fallback_backend") or "marker")
    write_manifest(
        output / "adapter_run.json",
        {
            "method": "PosterMELD",
            "upstream": "Shannon4Science/PosterMELD",
            "upstream_commit": "0281595",
            "strict_upstream": not bool(rejected_draft_report.get("triggered")),
            "only_adaptation": (
                "codex-exec transport; precomputed MinerU pipeline input; retain first upstream poster artifact without quality filtering"
                if precomputed_bundle is not None
                else "codex-exec transport; retain first upstream poster artifact without quality filtering"
            ),
            "status": status,
            "error": error,
            "upstream_exit_code": upstream_exit_code,
            "poster_artifact_stage": artifact_stage,
            "upstream_result_status": (
                "rejected" if rejected_draft_report.get("triggered") else "completed"
            ),
            "upstream_rejected_draft_exported": bool(rejected_draft_report.get("triggered")),
            "upstream_rejection_reason": rejected_draft_report.get("rejection_reason"),
            "result_acceptance_policy": (
                "export_upstream_rejected_draft_without_repair_or_resampling"
                if rejected_draft_report.get("triggered")
                else "first_available_output_no_quality_filter"
            ),
            "paper": str(paper),
            "input_policy": (
                "precomputed_mineru_pipeline_with_upstream_asset_parser"
                if precomputed_bundle is not None
                else "upstream_mineru_with_official_marker_fallback"
            ),
            "mineru_base_url": None if precomputed_bundle is not None else args.mineru_base_url,
            "mineru_transport": (
                "precomputed_pipeline_cache"
                if precomputed_bundle is not None
                else (
                    "local_vlm_cloud_v4_bridge"
                    if args.mineru_base_url.startswith("http://127.0.0.1:")
                    else "remote_cloud_v4"
                )
            ),
            "mineru_precomputed_bundle": str(precomputed_bundle) if precomputed_bundle is not None else None,
            "mineru_endpoint_metadata": mineru_endpoint_metadata,
            "parser_backend": parser_backend,
            "parser_report": parser_report,
            "marker_pdf_version": _package_version("marker-pdf"),
            "marker_pdf_source": _direct_url_metadata("marker-pdf"),
            "model_backend": "codex exec --ephemeral",
            "model": args.model,
            "reasoning_effort": args.effort,
            "benchmark_profile": {
                "layout_template": "auto",
                "poster_style": "navy_serif",
                "visual_density": "balanced",
                "section_title_numbering": "off",
                "header_seed": 20260710,
                "vlm_layout_review": True,
                "visual_legibility_review": True,
                "block_vlm_review": True,
                "adaptive_column_width": True,
                "generated_background": False,
                "generated_teaser": False,
                "affiliation_logo_policy": affiliation_logo_policy,
                "conference": args.conference,
                "explicit_poster_size": (
                    [args.poster_width, args.poster_height]
                    if args.poster_width is not None
                    else None
                ),
                "recursion_limit": 64,
            },
            "quality_gate_accepted": (
                bool(quality_gate.get("accepted")) if quality_gate else None
            ),
            "quality_gate_failures": quality_gate.get("failures", []),
            "total_wall_seconds": wall_seconds,
            "api_equivalent_cost_usd": usage["estimated_api_cost_usd"],
            "codex_usage": usage,
            "post_mineru_generation_seconds": post_mineru_generation_seconds,
            "mineru_parse_time_seconds": (
                REFERENCE_MINERU_SUCCESS_MEAN_SECONDS
                if precomputed_bundle is not None
                else None
            ),
            "mineru_parse_time_kind": (
                "reference_success_mean_estimate"
                if precomputed_bundle is not None
                else None
            ),
            "mineru_batch_total_seconds": (
                REFERENCE_MINERU_BATCH_TOTAL_SECONDS
                if precomputed_bundle is not None
                else None
            ),
            "poster_png": str(output / "poster.png")
            if (output / "poster.png").is_file()
            else None,
            "poster_pptx": str(output / "poster.pptx")
            if (output / "poster.pptx").is_file()
            else None,
            "retained_workdir": str(workdir)
            if status != "completed" or args.keep_workdir
            else None,
        },
    )
    if status == "completed" and not args.keep_workdir:
        shutil.rmtree(workdir)
    if error:
        print(f"PosterMELD failed: {error}", file=sys.stderr)
    else:
        quality_note = (
            f", quality_gate={quality_gate.get('accepted')}"
            if quality_gate
            else ""
        )
        print(
            f"PosterMELD completed: {wall_seconds:.1f}s, "
            f"calls={usage['call_count']}, "
            f"api-equivalent=${usage['estimated_api_cost_usd']:.4f}"
            f"{quality_note}"
        )
    return 0 if status == "completed" else 1


def main(argv: Sequence[str] | None = None) -> int:
    return run(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
