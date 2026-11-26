"""Precompute P2P's native four-task DocLayout detector caches.

The output files are byte-for-byte compatible with the cache names consumed by
``third_party/P2P/main.py``. No model or layout stage is invoked here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_ROOT = REPO_ROOT / "third_party" / "P2P"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("cache_root", type=Path)
    parser.add_argument("--detector-url", default="http://127.0.0.1:8019/")
    parser.add_argument("--limit", type=int, default=100)
    return parser


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _valid(path: Path) -> bool:
    try:
        return isinstance(json.loads(path.read_text(encoding="utf-8")), list)
    except (OSError, json.JSONDecodeError):
        return False


def main() -> int:
    args = _parser().parse_args()
    papers = sorted(args.dataset_root.expanduser().resolve().glob("*/paper.pdf"))[: args.limit]
    if len(papers) != args.limit:
        raise SystemExit(f"Expected {args.limit} papers, found {len(papers)}")
    if str(UPSTREAM_ROOT) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_ROOT))
    from poster.figures import extract_figures

    cache_root = args.cache_root.expanduser().resolve()
    for index, paper in enumerate(papers, 1):
        output = cache_root / paper.parent.name
        figures_path = output / "paper_figures.json"
        captions_path = output / "paper_figures_cap.json"
        if _valid(figures_path) and _valid(captions_path):
            print(f"[{index:03d}/{len(papers)}] cached {paper.parent.name}", flush=True)
            continue
        started = time.monotonic()
        figure_items = extract_figures(args.detector_url, str(paper), task="figure")
        table_items = extract_figures(args.detector_url, str(paper), task="table")
        figure_captions = extract_figures(args.detector_url, str(paper), task="figurecaption")
        table_captions = extract_figures(args.detector_url, str(paper), task="tablecaption")
        threshold = 0.85
        while True:
            figures = [image for image, score in figure_items + table_items if score >= threshold]
            captions = [
                image
                for image, score in figure_captions + table_captions
                if score >= threshold
            ]
            if len(figures) == len(captions):
                break
            threshold -= 0.05
        _atomic_json(figures_path, figures)
        _atomic_json(captions_path, captions)
        _atomic_json(
            output / "preprocess.json",
            {
                "paper": str(paper),
                "detector_url": args.detector_url,
                "figure_count": len(figures),
                "caption_count": len(captions),
                "threshold": round(threshold, 4),
                "duration_seconds": round(time.monotonic() - started, 3),
            },
        )
        print(
            f"[{index:03d}/{len(papers)}] generated figures={len(figures)} "
            f"threshold={threshold:.2f} {paper.parent.name}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
