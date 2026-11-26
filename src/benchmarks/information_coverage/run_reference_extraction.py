"""Run official-poster information-unit extraction for any IDs in 0-99."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
import benchmark_common as common
import extract_remaining90 as extraction


PORTABLE_MANIFEST = common.BENCH_ROOT / "portable_manifest.json"
REFERENCE_PATH = common.BENCH_ROOT / "reference_information_units_full100.json"


def load_manifest() -> list[dict[str, Any]]:
    rows = common.read_json(PORTABLE_MANIFEST)["posters"]
    if [str(row["poster_id"]) for row in rows] != [str(i) for i in range(100)]:
        raise RuntimeError("Portable manifest IDs are not exactly 0-99")
    return rows


def aggregate(rows: list[dict[str, Any]]) -> None:
    posters = []
    for row in rows:
        result = common.read_json(
            extraction.RESULT_ROOT / f"{row['poster_id']}.json"
        )
        common.validate_units(row, result)
        normalized = dict(result)
        normalized.pop("generated_poster", None)
        normalized.update(
            {
                key: row[key]
                for key in ("official_poster", "rs_poster", "whh_poster")
            }
        )
        posters.append(normalized)
    common.write_json(
        REFERENCE_PATH,
        {
            "benchmark_name": "official-poster information units, full 100",
            "reference_source": "official poster images only; no paper text used",
            "unit_policy": {"units_per_poster": 7, "total_units": 700},
            "poster_count": 100,
            "posters": posters,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ids", nargs="*", type=int)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    all_rows = load_manifest()
    selected_ids = set(args.ids) if args.ids else set(range(100))
    if any(value < 0 or value > 99 for value in selected_ids):
        raise ValueError("--ids values must be between 0 and 99")
    selected = [row for row in all_rows if int(row["poster_id"]) in selected_ids]

    extraction.RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    extraction.LOG_ROOT.mkdir(parents=True, exist_ok=True)
    codex_prefix = common.resolve_codex()
    template = extraction.PROMPT_PATH.read_text(encoding="utf-8")
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                extraction.run_one,
                codex_prefix,
                template,
                row,
                args.force,
            ): row
            for row in selected
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                record = future.result()
                print(
                    f"[{row['poster_id']}] {record['status']} "
                    f"attempt={record.get('attempt', '-')}"
                )
            except Exception as exc:  # preserve other completed independent runs
                message = f"{row['poster_id']}: {type(exc).__name__}: {exc}"
                failures.append(message)
                print(f"[{row['poster_id']}] FAILED: {exc}")
    if failures:
        common.write_json(
            common.BENCH_ROOT / "reference_extraction" / "failures.json",
            {"failures": failures},
        )
        raise RuntimeError("; ".join(failures))
    aggregate(all_rows)
    print("Wrote reference_information_units_full100.json")


if __name__ == "__main__":
    main()
