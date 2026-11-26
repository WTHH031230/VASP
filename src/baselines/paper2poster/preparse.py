"""GPU-only Paper2Poster PDF preprocessing.

Produces the exact Markdown and figure/table artifacts consumed by the upstream
pipeline without making any LLM/VLM calls.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from .run import (
    DEFAULT_UPSTREAM,
    REPOSITORY_ROOT,
    _git_revision,
    _install_camel_shims,
    _install_namespace,
    _prepare_workdir,
    _slug,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-workdir", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_complete(cache: Path) -> bool:
    manifest = cache / "preparse.json"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        data.get("status") == "complete"
        and (cache / "markdown.txt").is_file()
        and (cache / "images_and_tables").is_dir()
    )


def main() -> int:
    args = _parser().parse_args()
    paper = args.paper.expanduser().resolve()
    cache = args.cache_dir.expanduser().resolve()
    upstream = args.upstream_root.expanduser().resolve()
    if not paper.is_file() or paper.suffix.lower() != ".pdf":
        raise SystemExit(f"Invalid paper PDF: {paper}")
    if not (upstream / "PosterAgent" / "parse_raw.py").is_file():
        raise SystemExit(f"Invalid Paper2Poster checkout: {upstream}")
    if _is_complete(cache) and not args.force:
        print(f"Preparse cache already complete: {cache}")
        return 0
    if cache.exists() and any(cache.iterdir()) and not args.force:
        raise SystemExit(f"Cache directory is non-empty: {cache}")
    if cache.exists() and args.force:
        shutil.rmtree(cache)

    cache.parent.mkdir(parents=True, exist_ok=True)
    slug = _slug(paper.parent.name)
    workdir = Path(tempfile.mkdtemp(prefix=f".preparse-{slug}-", dir=cache.parent))
    temporary = Path(tempfile.mkdtemp(prefix=f".{cache.name}.publish-", dir=cache.parent))
    succeeded = False
    original_cwd = Path.cwd()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        relative_paper = _prepare_workdir(
            workdir,
            upstream,
            paper,
            slug,
            "current",
            "paper-v2",
        )
        os.chdir(workdir)
        sys.path.insert(0, str(upstream))
        _install_camel_shims()
        _install_namespace("PosterAgent", upstream / "PosterAgent")
        _install_namespace("utils", upstream / "utils")
        parse_module = importlib.import_module("PosterAgent.parse_raw")

        gpu_started = time.monotonic()
        convert_started = time.monotonic()
        conversion = parse_module.doc_converter.convert(relative_paper.as_posix())
        docling_seconds = time.monotonic() - convert_started

        raw_markdown = conversion.document.export_to_markdown()
        cleaned = re.sub(r"<!--[\s\S]*?-->", "", raw_markdown)
        markdown_for_generation = raw_markdown
        marker_fallback = False
        if len(cleaned) < 500:
            marker_fallback = True
            parser_model = parse_module.create_model_dict(
                device="cuda",
                dtype=parse_module.torch.float16,
            )
            markdown_for_generation, _rendered = parse_module.parse_pdf(
                relative_paper.as_posix(),
                model_lst=parser_model,
                save_file=False,
            )

        export_started = time.monotonic()
        upstream_args = SimpleNamespace(
            poster_path=relative_paper.as_posix(),
            model_name_t="codex",
            model_name_v="codex",
            poster_name=slug,
        )
        parse_module.gen_image_and_table(upstream_args, conversion)
        asset_export_seconds = time.monotonic() - export_started
        gpu_preparse_seconds = time.monotonic() - gpu_started

        source = workdir / "<codex_codex>_images_and_tables"
        if not source.is_dir():
            raise RuntimeError(f"Missing exported image/table directory: {source}")
        shutil.copytree(source, temporary / "images_and_tables")
        (temporary / "markdown.txt").write_text(
            str(markdown_for_generation),
            encoding="utf-8",
        )
        manifest = {
            "status": "complete",
            "paper": str(paper),
            "paper_sha256": _sha256(paper),
            "slug": slug,
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "gpu_preparse_seconds": round(gpu_preparse_seconds, 2),
            "docling_seconds": round(docling_seconds, 2),
            "asset_export_seconds": round(asset_export_seconds, 2),
            "marker_fallback": marker_fallback,
            "markdown_characters": len(str(markdown_for_generation)),
            "upstream_root": str(upstream),
            "upstream_revision": _git_revision(upstream),
            "llm_calls": 0,
        }
        (temporary / "preparse.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if cache.exists():
            cache.rmdir()
        temporary.replace(cache)
        succeeded = True
        print(
            f"GPU preparse complete: {paper.parent.name} in "
            f"{gpu_preparse_seconds:.2f}s -> {cache}",
            flush=True,
        )
        return 0
    finally:
        os.chdir(original_cwd)
        if succeeded and not args.keep_workdir:
            shutil.rmtree(workdir)
        elif workdir.exists():
            print(f"Preparse workdir retained at: {workdir}", file=sys.stderr)
        if temporary.exists():
            shutil.rmtree(temporary)


if __name__ == "__main__":
    raise SystemExit(main())
