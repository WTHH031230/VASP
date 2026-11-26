"""Run upstream P2P unchanged except for its chat-model transport.

The strict adapter deliberately uses P2P's own four-task DocLayout detector
service and its separate figure/caption crops. Only OpenAI chat constructors
are replaced with the repository's isolated ``codex exec`` backend.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from src.baselines.common.codex import CodexRuntime, configure_codex, get_codex_run_stats
from src.baselines.common.manifest import write_manifest
from src.baselines.p2p.langchain_backend import install


REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_ROOT = REPO_ROOT / "third_party" / "P2P"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--detector-url",
        default=os.getenv("P2P_FIGURE_DETECTOR_URL", ""),
        help="Official figure_detection.py POST endpoint (or P2P_FIGURE_DETECTOR_URL)",
    )
    parser.add_argument(
        "--detector-cache-dir",
        type=Path,
        help=(
            "optional precomputed directory containing paper_figures.json and "
            "paper_figures_cap.json; these are copied beside the temporary PDF "
            "so P2P consumes its own native cache format"
        ),
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--keep-workdir", action="store_true")
    return parser.parse_args()


def _copy_detector_cache(input_pdf: Path, output_dir: Path) -> list[str]:
    cache_dir = output_dir / "detector_cache"
    copied: list[str] = []
    for suffix in ("_figures.json", "_figures_cap.json"):
        source = Path(str(input_pdf).removesuffix(".pdf") + suffix)
        if not source.is_file():
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / source.name
        shutil.copy2(source, target)
        copied.append(str(target))
    return copied


def _seed_detector_cache(input_pdf: Path, cache_dir: Path | None) -> list[str]:
    if cache_dir is None:
        return []
    resolved = cache_dir.expanduser().resolve()
    seeded: list[str] = []
    for suffix in ("_figures.json", "_figures_cap.json"):
        source = resolved / f"paper{suffix}"
        if not source.is_file():
            raise FileNotFoundError(f"Missing P2P detector cache: {source}")
        target = Path(str(input_pdf).removesuffix(".pdf") + suffix)
        shutil.copy2(source, target)
        seeded.append(str(source))
    return seeded


def _detector_metadata(detector_url: str) -> dict[str, object]:
    """Read local audit metadata without requiring it from upstream Modal."""

    health_url = detector_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=5) as response:
            payload = json.load(response)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def run(args: argparse.Namespace) -> int:
    paper = args.paper.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not paper.is_file():
        raise SystemExit(f"Paper not found: {paper}")
    if not args.detector_url.strip():
        raise SystemExit(
            "Strict P2P requires the official DocLayout detector endpoint; "
            "pass --detector-url or set P2P_FIGURE_DETECTOR_URL."
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to replace non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=".p2p-strict-", dir=output_dir.parent))
    input_dir = workdir / paper.parent.name
    input_dir.mkdir(parents=True)
    input_pdf = input_dir / "paper.pdf"
    shutil.copy2(paper, input_pdf)
    seeded_detector_cache = _seed_detector_cache(input_pdf, args.detector_cache_dir)
    trace_dir = output_dir / "codex_traces"
    detector_metadata = _detector_metadata(args.detector_url)
    configure_codex(
        CodexRuntime(
            executable=args.codex_bin,
            model=args.model,
            reasoning_effort=args.effort,
            timeout_s=args.timeout,
            cwd=UPSTREAM_ROOT,
            trace_dir=trace_dir,
            application="p2p-strict",
            log_prefix="p2p-strict/codex",
            result_dir=output_dir,
        )
    )

    if str(UPSTREAM_ROOT) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_ROOT))
    from poster import poster as poster_module

    install(poster_module)
    import main as upstream_main

    json_path = output_dir / "poster.json"
    html_path = output_dir / "poster.html"
    png_path = output_dir / "poster.png"
    started = time.monotonic()
    status = "failed"
    error: str | None = None
    try:
        # Official entry point: four detector tasks, threshold pairing, separate
        # caption crops, and upstream generation/layout retry behavior.
        upstream_main.generate_paper_poster(
            url=args.detector_url,
            pdf=str(input_pdf),
            vendor="openai",
            model=args.model,
            text_prompt="",
            figures_prompt="",
            output=str(json_path),
        )
        status = "completed" if png_path.is_file() else "failed"
        if status == "failed":
            error = "P2P completed without poster.png"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    upstream_error = error
    if png_path.is_file():
        status = "completed"
        error = None
    elif status == "completed":
        status = "failed"
        error = "P2P completed without poster.png"

    detector_cache = _copy_detector_cache(input_pdf, output_dir)
    wall_seconds = round(time.monotonic() - started, 3)
    usage = get_codex_run_stats()
    write_manifest(
        output_dir / "adapter_run.json",
        {
            "method": "P2P",
            "upstream": "multimodal-art-projection/P2P",
            "upstream_commit": "c619c9c",
            "strict_upstream": True,
            "only_adaptation": "OpenAI chat model -> codex exec --ephemeral",
            "status": status,
            "error": error,
            "upstream_error": upstream_error,
            "result_acceptance_policy": "first_available_output_no_quality_filter",
            "paper": str(paper),
            "input_policy": "upstream_four_task_doclayout_detector",
            "detector_url": args.detector_url,
            "detector_metadata": detector_metadata,
            "detector_cache": detector_cache,
            "seeded_detector_cache": seeded_detector_cache,
            "model_backend": "codex exec --ephemeral",
            "model": args.model,
            "reasoning_effort": args.effort,
            "total_wall_seconds": wall_seconds,
            "api_equivalent_cost_usd": usage["estimated_api_cost_usd"],
            "codex_usage": usage,
            "poster_png": str(png_path) if png_path.is_file() else None,
            "poster_html": str(html_path) if html_path.is_file() else None,
            "poster_json": str(json_path) if json_path.is_file() else None,
            "retained_workdir": str(workdir) if error or args.keep_workdir else None,
        },
    )
    if not error and not args.keep_workdir:
        shutil.rmtree(workdir)
    if error:
        print(f"P2P failed: {error}", file=sys.stderr)
    else:
        print(
            f"P2P completed: {wall_seconds:.1f}s, calls={usage['call_count']}, "
            f"api-equivalent=${usage['estimated_api_cost_usd']:.4f}"
        )
    return 0 if status == "completed" else 1


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
