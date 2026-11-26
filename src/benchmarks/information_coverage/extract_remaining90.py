from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
import benchmark_common as common


SCHEMA_PATH = common.BENCH_ROOT.parents[2] / "configs" / "schemas" / "information_coverage" / "extraction_output_schema.json"
PROMPT_PATH = common.BENCH_ROOT / "extraction_prompt.md"
STAGE_ROOT = common.BENCH_ROOT / "reference_extraction"
RESULT_ROOT = STAGE_ROOT / "results"
LOG_ROOT = STAGE_ROOT / "logs"
FIRST10_REFERENCE = (
    common.BENCH_ROOT.parent / "first10_rs" / "reference_information_units.json"
)


def build_prompt(template: str, poster: dict[str, Any]) -> str:
    payload = {
        "poster_id": poster["poster_id"],
        "directory_title_hint": poster["directory_title"],
    }
    return template.rstrip() + "\n\nINPUT:\n" + json.dumps(
        payload, ensure_ascii=False, indent=2
    )


def run_one(
    codex_prefix: list[str],
    template: str,
    poster: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    poster_id = poster["poster_id"]
    result_path = RESULT_ROOT / f"{poster_id}.json"
    if result_path.is_file() and not force:
        existing = common.read_json(result_path)
        common.validate_units(poster, existing)
        return {"poster_id": poster_id, "status": "reused", "attempt": None}

    staged_image = STAGE_ROOT / "staged_official_images" / f"{poster_id}.png"
    image_path = staged_image if staged_image.is_file() else common.ROOT / poster["official_poster"]
    prompt = build_prompt(template, poster)
    poster_log_root = LOG_ROOT / poster_id
    common.clear_attempt_artifacts(poster_log_root, force)
    errors: list[str] = []
    for attempt in range(1, 3):
        last_message = poster_log_root / f"attempt_{attempt}_last_message.json"
        returncode, elapsed, _ = common.run_codex(
            codex_prefix=codex_prefix,
            prompt=prompt,
            image_path=image_path,
            schema_path=SCHEMA_PATH,
            last_message=last_message,
            log_path=poster_log_root / f"attempt_{attempt}.json",
            poster_id=poster_id,
            stage="reference_extraction",
            attempt=attempt,
        )
        try:
            if returncode != 0:
                raise RuntimeError(f"codex exec returned {returncode}")
            if not last_message.is_file():
                raise RuntimeError("missing --output-last-message")
            result = common.read_json(last_message)
            common.validate_units(poster, result)
            result["official_poster"] = poster["official_poster"]
            result["rs_poster"] = poster["rs_poster"]
            result["whh_poster"] = poster["whh_poster"]
            result["metadata"] = {
                "source": "official poster image only",
                "model": common.MODEL,
                "reasoning_effort": common.REASONING_EFFORT,
                "session_policy": "fresh ephemeral codex exec; never resumed",
                "attempt": attempt,
            }
            common.write_json(result_path, result)
            return {
                "poster_id": poster_id,
                "status": "completed",
                "attempt": attempt,
                "elapsed_seconds": elapsed,
            }
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            common.write_json(
                poster_log_root / f"attempt_{attempt}_error.json",
                {"error": errors[-1]},
            )
    raise RuntimeError(f"poster {poster_id} failed: {'; '.join(errors)}")


def aggregate(posters: list[dict[str, Any]]) -> None:
    remaining: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for poster in posters:
        result = common.read_json(RESULT_ROOT / f"{poster['poster_id']}.json")
        common.validate_units(poster, result)
        remaining.append(result)
        records.append(
            {
                "poster_id": poster["poster_id"],
                "status": "completed",
                "attempt": result.get("metadata", {}).get("attempt"),
            }
        )
    common.write_json(
        common.BENCH_ROOT / "reference_information_units_10_99.json",
        {
            "benchmark_name": "official-poster information units, posters 10-99",
            "reference_source": "official poster images only; no paper text used",
            "unit_policy": {"units_per_poster": 7},
            "poster_count": 90,
            "posters": remaining,
            "run_records": sorted(records, key=lambda item: int(item["poster_id"])),
        },
    )

    first10 = common.read_json(FIRST10_REFERENCE)["posters"]
    if [item["poster_id"] for item in first10] != [str(i) for i in range(10)]:
        raise RuntimeError("first-10 reference file has unexpected IDs")
    full = first10 + remaining
    if [item["poster_id"] for item in full] != [str(i) for i in range(100)]:
        raise RuntimeError("full reference aggregation is not 0-99")
    common.write_json(
        common.BENCH_ROOT / "reference_information_units_full100.json",
        {
            "benchmark_name": "official-poster information units, full 100",
            "reference_source": "official poster images only; no paper text used",
            "unit_policy": {"units_per_poster": 7, "total_units": 700},
            "poster_count": 100,
            "posters": full,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--ids",
        nargs="*",
        type=int,
        help="Optional subset of poster IDs to run; aggregation still validates all 10-99 results.",
    )
    args = parser.parse_args()
    manifest = common.ensure_manifest()
    all_posters = [item for item in manifest if 10 <= int(item["poster_id"]) <= 99]
    selected_ids = set(args.ids) if args.ids else set(range(10, 100))
    if any(value < 10 or value > 99 for value in selected_ids):
        raise ValueError("--ids values must be between 10 and 99")
    posters = [item for item in all_posters if int(item["poster_id"]) in selected_ids]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    failure_path = STAGE_ROOT / "failures.json"
    if args.force and failure_path.exists():
        failure_path.unlink()
    codex_prefix = common.resolve_codex()
    template = PROMPT_PATH.read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, codex_prefix, template, poster, args.force): poster
            for poster in posters
        }
        for future in as_completed(futures):
            poster = futures[future]
            try:
                record = future.result()
                records.append(record)
                print(
                    f"[{poster['poster_id']}] {record['status']} "
                    f"attempt={record.get('attempt', '-')}"
                )
            except Exception as exc:
                message = f"{poster['poster_id']}: {type(exc).__name__}: {exc}"
                failures.append(message)
                print(f"[{poster['poster_id']}] FAILED: {exc}")
    if failures:
        common.write_json(failure_path, {"failures": failures})
        raise RuntimeError("; ".join(failures))
    aggregate(all_posters)
    print("Wrote remaining-90 and full-100 reference information-unit files")


if __name__ == "__main__":
    main()
