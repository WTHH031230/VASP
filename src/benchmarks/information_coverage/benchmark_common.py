from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BENCH_ROOT = Path(__file__).resolve().parent
MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "medium"

DIRECTORY_TITLE_OVERRIDES = {
    # The directory retains its historical CoT-based name, but the supplied
    # poster and paper were corrected to the ICML 2024 ICL paper on 2026-08-09.
    "47": "How Do Nonlinear Transformers Learn and Generalize in In-Context Learning?",
    # The directory retains the later ICML 2024 paper title, but poster.png was
    # corrected to match the supplied ICLR 2022 paper on 2026-08-09.
    "65": "A Neural Tangent Kernel Perspective of Infinite Tree Ensembles",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def discover_manifest() -> list[dict[str, Any]]:
    official: dict[int, Path] = {}
    for directory in (ROOT / "papers").iterdir():
        if not directory.is_dir():
            continue
        match = re.match(r"^(\d+)_(.+)$", directory.name)
        if match:
            official[int(match.group(1))] = directory

    def discover_generated(variant: str) -> dict[int, Path]:
        found: dict[int, Path] = {}
        for path in (Path(os.environ.get("VASP_POSTER_RESULTS_DIR", str(ROOT / "results"))) / "only_poster" / variant).glob("*.png"):
            match = re.match(r"^(\d+)_", path.name)
            if match:
                found[int(match.group(1))] = path
        return found

    rs = discover_generated("rs")
    whh = discover_generated("whh")
    manifest: list[dict[str, Any]] = []
    for poster_id in range(100):
        if poster_id not in official or poster_id not in rs or poster_id not in whh:
            raise RuntimeError(f"Missing source mapping for poster {poster_id}")
        poster_path = official[poster_id] / "poster.png"
        if not poster_path.is_file():
            raise FileNotFoundError(poster_path)
        if rs[poster_id].name != whh[poster_id].name:
            raise RuntimeError(
                f"RS/WHH filename mismatch for {poster_id}: "
                f"{rs[poster_id].name} != {whh[poster_id].name}"
            )
        poster_id_text = str(poster_id)
        directory_title = DIRECTORY_TITLE_OVERRIDES.get(
            poster_id_text,
            re.sub(r"^\d+_", "", official[poster_id].name),
        )
        manifest.append(
            {
                "poster_id": poster_id_text,
                "directory_title": directory_title,
                "official_poster": rel(poster_path),
                "rs_poster": rel(rs[poster_id]),
                "whh_poster": rel(whh[poster_id]),
            }
        )
    return manifest


def ensure_manifest() -> list[dict[str, Any]]:
    manifest = discover_manifest()
    write_json(
        BENCH_ROOT / "manifest.json",
        {
            "poster_count": len(manifest),
            "mapping_policy": "numeric ID prefix; RS and WHH filenames must match",
            "posters": manifest,
        },
    )
    return manifest


def resolve_codex() -> list[str]:
    executable = shutil.which("codex")
    if not executable:
        raise RuntimeError("codex executable was not found on PATH")
    path = Path(executable)
    if os.name == "nt" and path.suffix.lower() in {".cmd", ".bat"}:
        codex_js = path.parent / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
        node = path.parent / "node.exe"
        node_executable = str(node) if node.is_file() else shutil.which("node")
        if not codex_js.is_file() or not node_executable:
            raise RuntimeError(f"Cannot resolve Windows Codex launcher safely: {path}")
        return [str(Path(node_executable).resolve()), str(codex_js.resolve())]
    return [str(path)]


def run_codex(
    *,
    codex_prefix: list[str],
    prompt: str,
    image_path: Path,
    schema_path: Path,
    last_message: Path,
    log_path: Path,
    poster_id: str,
    stage: str,
    attempt: int,
    timeout_seconds: int = 900,
) -> tuple[int, float, str]:
    command = [
        *codex_prefix,
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "-C",
        str(BENCH_ROOT),
        "-m",
        MODEL,
        "-c",
        f'model_reasoning_effort="{REASONING_EFFORT}"',
        "--image",
        str(image_path.resolve()),
        "--output-schema",
        str(schema_path.resolve()),
        "--output-last-message",
        str(last_message.resolve()),
        "-",
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=BENCH_ROOT,
        timeout=timeout_seconds,
        check=False,
    )
    elapsed = round(time.perf_counter() - started, 3)
    write_json(
        log_path,
        {
            "poster_id": poster_id,
            "stage": stage,
            "attempt": attempt,
            "command": command,
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "session_policy": "fresh ephemeral codex exec invocation; never resume",
            "image": str(image_path.resolve()),
            "prompt": prompt,
            "returncode": completed.returncode,
            "elapsed_seconds": elapsed,
            "combined_output": completed.stdout,
        },
    )
    return completed.returncode, elapsed, completed.stdout


def clear_attempt_artifacts(log_root: Path, force: bool) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    if not force:
        return
    for attempt in range(1, 3):
        for suffix in (".json", "_error.json", "_last_message.json"):
            path = log_root / f"attempt_{attempt}{suffix}"
            if path.exists():
                path.unlink()


def validate_units(poster: dict[str, Any], result: dict[str, Any]) -> None:
    poster_id = poster["poster_id"]
    if result.get("poster_id") != poster_id:
        raise ValueError(f"poster_id mismatch for {poster_id}")
    title = result.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"missing title for {poster_id}")
    units = result.get("information_units")
    if not isinstance(units, list) or len(units) != 7:
        raise ValueError(f"poster {poster_id} must have exactly seven units")
    expected_ids = [f"{poster_id}-U{i}" for i in range(1, 8)]
    actual_ids = [unit.get("unit_id") for unit in units]
    if actual_ids != expected_ids:
        raise ValueError(f"unit IDs mismatch for {poster_id}: {actual_ids}")
    fields = [
        "information_type",
        "topic",
        "local_context",
        "statement",
        "coverage_requirement",
        "official_evidence",
    ]
    for unit in units:
        for field in fields:
            if not isinstance(unit.get(field), str) or not unit[field].strip():
                raise ValueError(f"empty {field} in {unit.get('unit_id')}")
    normalized = [unit["statement"].strip().lower() for unit in units]
    if len(set(normalized)) != 7:
        raise ValueError(f"duplicate unit statements for {poster_id}")
    unavailable_patterns = [
        r"no poster(?:-| )?(?:supported|grounded|visible)",
        r"poster image is not visible",
        r"without the visible official poster",
        r"reattach the official poster",
        r"cannot be (?:reliably )?extracted from the unavailable image",
        r"no core information unit can be",
        r"no .*information unit can be (?:determined|identified|established)",
    ]
    for unit in units:
        if unit["information_type"].strip().lower() == "unavailable":
            raise ValueError(f"unavailable-image placeholder in {unit['unit_id']}")
        combined = " ".join(
            str(unit[field])
            for field in ("statement", "coverage_requirement", "official_evidence")
        ).lower()
        if any(re.search(pattern, combined) for pattern in unavailable_patterns):
            raise ValueError(f"unavailable-image placeholder in {unit['unit_id']}")


def validate_evaluation(poster: dict[str, Any], result: dict[str, Any]) -> None:
    poster_id = poster["poster_id"]
    if result.get("poster_id") != poster_id:
        raise ValueError(f"poster_id mismatch for {poster_id}")
    assessments = result.get("assessments")
    if not isinstance(assessments, list) or len(assessments) != 7:
        raise ValueError(f"poster {poster_id} must have seven assessments")
    expected_ids = [unit["unit_id"] for unit in poster["information_units"]]
    actual_ids = [item.get("unit_id") for item in assessments]
    if actual_ids != expected_ids:
        raise ValueError(f"assessment IDs mismatch for {poster_id}: {actual_ids}")
    for item in assessments:
        if item.get("verdict") not in {"COVERED", "NOT_COVERED"}:
            raise ValueError(f"invalid verdict in {item.get('unit_id')}")
        for field in ("evidence", "rationale"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"empty {field} in {item.get('unit_id')}")
    computed = sum(item["verdict"] == "NOT_COVERED" for item in assessments)
    if result.get("missing_count") != computed:
        raise ValueError(f"missing_count mismatch for {poster_id}")
