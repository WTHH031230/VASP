"""Run official, RS, and WHH information-coverage evaluation for IDs 0-99."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
import benchmark_common as common
import consolidate_full100 as consolidation
import evaluate_remaining90 as evaluation


PORTABLE_MANIFEST = common.BENCH_ROOT / "portable_manifest.json"
REFERENCE_PATH = common.BENCH_ROOT / "reference_information_units_full100.json"


def load_posters() -> list[dict[str, Any]]:
    posters = common.read_json(REFERENCE_PATH)["posters"]
    rows = common.read_json(PORTABLE_MANIFEST)["posters"]
    if [str(row["poster_id"]) for row in rows] != [str(i) for i in range(100)]:
        raise RuntimeError("Portable manifest IDs are not exactly 0-99")
    mapped = {str(row["poster_id"]): row for row in rows}
    if [str(poster["poster_id"]) for poster in posters] != [
        str(i) for i in range(100)
    ]:
        raise RuntimeError("Reference IDs are not exactly 0-99")
    for poster in posters:
        common.validate_units(poster, poster)
        poster.update(
            {
                key: mapped[str(poster["poster_id"])][key]
                for key in ("official_poster", "rs_poster", "whh_poster")
            }
        )
    return posters


def run_variant(
    variant: str,
    posters: list[dict[str, Any]],
    selected_ids: set[int],
    workers: int,
    force: bool,
) -> None:
    stage_root = common.BENCH_ROOT / "evaluations" / variant
    result_root = stage_root / "results"
    log_root = stage_root / "logs"
    result_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    codex_prefix = common.resolve_codex()
    template = evaluation.PROMPTS[variant].read_text(encoding="utf-8")
    selected = [
        poster for poster in posters if int(poster["poster_id"]) in selected_ids
    ]
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluation.run_one,
                codex_prefix,
                template,
                poster,
                variant,
                result_root,
                log_root,
                force,
            ): poster
            for poster in selected
        }
        for future in as_completed(futures):
            poster = futures[future]
            try:
                record = future.result()
                print(
                    f"[{variant}:{poster['poster_id']}] {record['status']} "
                    f"missing={record['missing_count']} "
                    f"attempt={record.get('attempt', '-')}"
                )
            except Exception as exc:  # preserve other completed independent runs
                message = f"{poster['poster_id']}: {type(exc).__name__}: {exc}"
                failures.append(message)
                print(f"[{variant}:{poster['poster_id']}] FAILED: {exc}")
    if failures:
        common.write_json(stage_root / "failures.json", {"failures": failures})
        raise RuntimeError("; ".join(failures))
    consolidation.create_variant_summary(variant, posters)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("official", "rs", "whh"),
        default=("official", "rs", "whh"),
    )
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ids", nargs="*", type=int)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    selected_ids = set(args.ids) if args.ids else set(range(100))
    if any(value < 0 or value > 99 for value in selected_ids):
        raise ValueError("--ids values must be between 0 and 99")
    posters = load_posters()
    for variant in args.variants:
        run_variant(variant, posters, selected_ids, args.workers, args.force)
    print("Evaluation summaries updated")


if __name__ == "__main__":
    main()
