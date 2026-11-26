#!/usr/bin/env python3
"""Run unmodified upstream PaperX (Paper2DAG + DAG2Poster) via Codex CLI (gpt-5.6-sol, medium)."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import types as pytypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

P2P_REPO = Path(os.environ.get("VASP_REPO_ROOT", str(Path(__file__).resolve().parents[3])))
EXP_DIR = Path(os.environ.get("VASP_OUT_DIR", "./outputs/paperx"))
PAPERX_ROOT = EXP_DIR / "PaperX"
INDEX_MAP_PATH = EXP_DIR / "paper_index_map.json"

# 1. Load shared Codex backend from p2p-research before importing PaperX src/
if str(P2P_REPO) not in sys.path:
    sys.path.insert(0, str(P2P_REPO))
from src.baselines.common.codex import (  # noqa: E402
    CodexRuntime,
    configure_codex,
    core as codex_core,
    get_codex_run_stats,
)

for _mod_name in list(sys.modules):
    if _mod_name == "src" or _mod_name.startswith("src."):
        del sys.modules[_mod_name]
if str(P2P_REPO) in sys.path:
    sys.path.remove(str(P2P_REPO))


# 2. Install non-invasive shims for google.genai and openai.OpenAI -> Codex CLI
@dataclass
class _FakePart:
    kind: str
    text: str = ""
    data: bytes = b""
    mime_type: str = "image/png"

    @classmethod
    def from_text(cls, text: str) -> "_FakePart":
        return cls(kind="text", text=str(text))

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str = "image/png") -> "_FakePart":
        return cls(kind="bytes", data=bytes(data), mime_type=mime_type)


class _FakeGenerateContentConfig:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass
class _FakeGenAIResponse:
    text: str


class _FakeGenAIModels:
    def generate_content(
        self,
        *,
        model: str,
        contents: Any,
        config: Any = None,
        **_: Any,
    ) -> _FakeGenAIResponse:
        del model
        sys_instr: str | None = None
        resp_mime: str | None = None
        if isinstance(config, dict):
            sys_instr = config.get("system_instruction")
            resp_mime = config.get("response_mime_type")
        elif config is not None:
            sys_instr = getattr(config, "system_instruction", None)
            resp_mime = getattr(config, "response_mime_type", None)

        dev_parts: list[str] = []
        if sys_instr:
            dev_parts.append(str(sys_instr))
        if resp_mime == "application/json":
            dev_parts.append("Output valid JSON only, without extra commentary.")
        dev_instructions = "\n\n".join(dev_parts) if dev_parts else None

        text_parts: list[str] = []
        images: list[Image.Image] = []
        items = contents if isinstance(contents, (list, tuple)) else [contents]
        try:
            for item in items:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, _FakePart):
                    if item.kind == "text":
                        text_parts.append(item.text)
                    elif item.kind == "bytes":
                        with Image.open(io.BytesIO(item.data)) as src_im:
                            images.append(src_im.convert("RGB").copy())
                elif isinstance(item, Image.Image):
                    images.append(item.convert("RGB").copy())
                else:
                    text_parts.append(str(item))
            prompt_str = "\n\n".join(text_parts)
            answer, _, _ = codex_core._invoke_codex(
                codex_core._runtime,
                prompt_str,
                images,
                developer_instructions=dev_instructions,
            )
            if resp_mime == "application/json" and answer:
                import re as _re
                answer = _re.sub(
                    r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
                    lambda m: f"\\u{ord(m.group(0)):04x}",
                    answer,
                )
            return _FakeGenAIResponse(text=answer)
        finally:
            for im in images:
                im.close()


class _FakeGenAIClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.models = _FakeGenAIModels()


@dataclass
class _FakeChatMessage:
    content: str


@dataclass
class _FakeChatChoice:
    message: _FakeChatMessage
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class _FakeChatCompletion:
    choices: list[_FakeChatChoice]


class _FakeChatCompletions:
    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **_: Any,
    ) -> _FakeChatCompletion:
        del model
        sys_parts: list[str] = []
        user_parts: list[str] = []
        images: list[Image.Image] = []
        try:
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    if isinstance(content, str):
                        sys_parts.append(content)
                    elif isinstance(content, list):
                        for blk in content:
                            if isinstance(blk, dict) and blk.get("type") == "text":
                                sys_parts.append(str(blk.get("text", "")))
                else:
                    if isinstance(content, str):
                        user_parts.append(content)
                    elif isinstance(content, list):
                        for blk in content:
                            if not isinstance(blk, dict):
                                continue
                            btype = blk.get("type")
                            if btype == "text":
                                user_parts.append(str(blk.get("text", "")))
                            elif btype == "image_url":
                                url = (blk.get("image_url") or {}).get("url", "")
                                if url.startswith("data:image/"):
                                    _, _, b64 = url.partition(",")
                                    raw = base64.b64decode(b64)
                                    with Image.open(io.BytesIO(raw)) as src_im:
                                        images.append(src_im.convert("RGB").copy())
            dev_instructions = "\n\n".join(p for p in sys_parts if p) or None
            prompt_str = "\n\n".join(p for p in user_parts if p)
            answer, _, _ = codex_core._invoke_codex(
                codex_core._runtime,
                prompt_str,
                images,
                developer_instructions=dev_instructions,
            )
            return _FakeChatCompletion(choices=[_FakeChatChoice(message=_FakeChatMessage(content=answer))])
        finally:
            for im in images:
                im.close()


class _FakeChatNamespace:
    def __init__(self) -> None:
        self.completions = _FakeChatCompletions()


class _FakeOpenAIClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.chat = _FakeChatNamespace()


def _install_shims_and_import_paperx():
    google_mod = sys.modules.get("google") or pytypes.ModuleType("google")
    genai_mod = pytypes.ModuleType("google.genai")
    types_mod = pytypes.ModuleType("google.genai.types")
    types_mod.GenerateContentConfig = _FakeGenerateContentConfig
    types_mod.Part = _FakePart
    genai_mod.Client = _FakeGenAIClient
    genai_mod.types = types_mod
    google_mod.genai = genai_mod
    sys.modules["google"] = google_mod
    sys.modules["google.genai"] = genai_mod
    sys.modules["google.genai.types"] = types_mod

    import openai
    openai.OpenAI = _FakeOpenAIClient

    if str(PAPERX_ROOT) not in sys.path:
        sys.path.insert(0, str(PAPERX_ROOT))
    paper2DAG = importlib.import_module("src.paper2DAG")
    DAG2poster = importlib.import_module("src.DAG2poster")
    refinement = importlib.import_module("src.refinement.refinement")
    return paper2DAG, DAG2poster, refinement


def run_single_paper(
    entry: dict[str, Any],
    *,
    model: str,
    effort: str,
    timeout_s: int,
    resume: bool,
) -> dict[str, Any]:
    idx = int(entry["index"])
    canon_name = str(entry["canonical_name"])
    auto_dir = Path(entry["staged_auto"])
    subdir_path = auto_dir.parent

    out_poster_dir = EXP_DIR / "posters" / canon_name
    out_poster_png = out_poster_dir / "poster.png"
    out_manifest = out_poster_dir / "manifest.json"
    eval_input_png = EXP_DIR / "eval" / "inputs" / "paperx" / f"{canon_name}.png"
    success_flag = auto_dir / "poster_success.txt"

    import fcntl
    out_poster_dir.mkdir(parents=True, exist_ok=True)
    lock_fh = (out_poster_dir / ".run.lock").open("w")
    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)

    if resume and success_flag.is_file() and out_poster_png.is_file() and out_manifest.is_file():
        print(f"[{idx:03d}/100] Reusing completed poster: {canon_name}", flush=True)
        return json.loads(out_manifest.read_text(encoding="utf-8"))

    paper2DAG, DAG2poster, refinement = _install_shims_and_import_paperx()

    trace_dir = EXP_DIR / "traces" / f"{idx:03d}"
    ledger_path = EXP_DIR / "logs" / "codex_ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    configure_codex(
        CodexRuntime(
            executable="codex",
            model=model,
            reasoning_effort=effort,
            timeout_s=timeout_s,
            max_retries=3,
            trace_dir=trace_dir,
            application="paperx",
            log_prefix=f"paperx/{idx:03d}",
            ledger_path=ledger_path,
            result_dir=out_poster_dir,
        )
    )

    prompts = json.loads((PAPERX_ROOT / "prompt.json").read_text(encoding="utf-8"))
    cfg = {
        "path": {"root_folder": str(EXP_DIR / "mineru_outputs")},
        "model_settings": {"generation_model": model},
        "api_keys": {"gemini_api_key": "codex-shim", "openai_api_key": "codex-shim"},
    }

    md_path = str(auto_dir / f"{canon_name}.md")
    pdf_path = str(auto_dir / f"{canon_name}_origin.pdf")
    if not (os.path.exists(md_path) and os.path.exists(pdf_path)):
        raise FileNotFoundError(f"Missing staged MinerU files in {auto_dir}")

    # Clean stale intermediate directories from any interrupted previous run
    section_split_dir = subdir_path / "section_split_output"
    section_dag_dir = subdir_path / "section_dag"
    for d in (section_split_dir, section_dag_dir):
        if d.exists():
            shutil.rmtree(d)

    t0 = time.monotonic()
    print(f"\n🚀 [{idx:03d}/100] Starting PaperX (model={model}, effort={effort}): {canon_name}", flush=True)

    # === Stage 1: Paper2DAG (exact main.py lines 101-132) ===
    cleaned_md_path = None
    for attempt in range(1, 4):
        cleaned_md_path = paper2DAG.clean_paper(md_path, prompts["clean_prompt"], model=model, config=cfg)
        if cleaned_md_path and os.path.isfile(cleaned_md_path) and os.path.getsize(cleaned_md_path) > 50:
            break
        print(f"⚠️ clean_paper attempt {attempt} returned empty; retrying...", flush=True)
    if not cleaned_md_path or not os.path.isfile(cleaned_md_path):
        raise RuntimeError(f"clean_paper failed for {canon_name}")

    split_paths = []
    for attempt in range(1, 4):
        if section_split_dir.exists():
            shutil.rmtree(section_split_dir)
        split_paths = paper2DAG.split_paper(cleaned_md_path, prompts["section_split_prompt"], model=model, config=cfg)
        if split_paths:
            break
        print(f"⚠️ split_paper attempt {attempt} returned 0 sections; retrying...", flush=True)
    if not split_paths:
        raise RuntimeError(f"split_paper produced 0 sections for {canon_name}")

    dag_path = str(auto_dir / "dag.json")
    paper2DAG.initialize_dag(
        markdown_path=cleaned_md_path,
        initialize_dag_prompt=prompts["initialize_dag_prompt"],
        model=model,
        config=cfg,
    )
    if not os.path.isfile(dag_path):
        raise RuntimeError(f"initialize_dag did not write {dag_path}")

    visual_dag_path = str(auto_dir / "visual_dag.json")
    paper2DAG.extract_and_generate_visual_dag(
        markdown_path=cleaned_md_path,
        prompt_for_gpt=prompts["visual_dag_prompt"],
        output_json_path=visual_dag_path,
        model=model,
        config=cfg,
    )
    paper2DAG.add_resolution_to_visual_dag(str(auto_dir), visual_dag_path)

    paper2DAG.build_section_dags(
        folder_path=str(section_split_dir),
        base_prompt=prompts["section_dag_generation_prompt"],
        model=model,
        config=cfg,
    )
    paper2DAG.add_section_dag(
        section_dag_folder=str(section_dag_dir),
        main_dag_path=dag_path,
        output_path=None,
    )
    paper2DAG.add_visual_dag(dag_path=dag_path, visual_dag_path=visual_dag_path)
    paper2DAG.refine_visual_node(dag_path)

    # === Stage 2: DAG2Poster (exact main.py lines 160-206) ===
    poster_outline_path = str(auto_dir / "poster_outline.txt")
    DAG2poster.generate_poster_outline_txt(
        dag_path=dag_path,
        poster_outline_path=poster_outline_path,
        poster_outline_prompt=prompts["poster_outline_prompt"],
        model=model,
        config=cfg,
    )

    poster_outline_modified = str(auto_dir / "poster_outline_modified.txt")
    DAG2poster.modify_poster_outline(
        poster_outline_path=poster_outline_path,
        poster_paper_name=canon_name,
        modified_poster_outline_path=poster_outline_modified,
    )

    backup_outline_mod = str(auto_dir / "poster_outline_modified.pre_logic.txt")
    shutil.copy2(poster_outline_modified, backup_outline_mod)
    last_logic_err: Exception | None = None
    for attempt in range(1, 4):
        shutil.copy2(backup_outline_mod, poster_outline_modified)
        try:
            DAG2poster.modified_poster_logic(
                poster_outline_modified,
                prompts["modified_poster_logic_prompt"],
                model=model,
                config=cfg,
            )
            last_logic_err = None
            break
        except Exception as exc:
            last_logic_err = exc
            print(f"⚠️ modified_poster_logic attempt {attempt}/3 error: {exc}", flush=True)
    if last_logic_err is not None:
        raise RuntimeError(f"modified_poster_logic failed after 3 attempts: {last_logic_err}") from last_logic_err

    poster_html_path = str(auto_dir / "poster.html")
    template_html = str(PAPERX_ROOT / "poster_template" / "poster_template.html")
    qr_banner_jpg = str(PAPERX_ROOT / "poster_template" / "expore_our_work_in_detail.jpg")

    DAG2poster.build_poster_from_outline(
        poster_outline_path=poster_outline_modified,
        poster_template_path=template_html,
        poster_path=poster_html_path,
    )
    DAG2poster.modify_title_and_author(dag_path=dag_path, poster_path=poster_html_path)

    final_dir = auto_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    poster_final_html = str(final_dir / "poster_final.html")
    poster_final_png = final_dir / "poster_final.png"
    if poster_final_png.exists():
        poster_final_png.unlink()

    DAG2poster.inject_img_section_to_poster(
        figure_path=qr_banner_jpg,
        auto_path=str(auto_dir),
        poster_path=poster_html_path,
    )

    for attempt in range(1, 4):
        if poster_final_png.exists():
            poster_final_png.unlink()
        refinement.refinement_poster(
            input_html_path=poster_html_path,
            prompts=prompts["poster_refinement_prompt"],
            output_html_path=poster_final_html,
            model=model,
            config=cfg,
        )
        if poster_final_png.is_file() and poster_final_png.stat().st_size > 5000:
            break
        print(f"⚠️ refinement_poster attempt {attempt}/3 did not produce valid poster_final.png; retrying...", flush=True)

    if not poster_final_png.is_file() or poster_final_png.stat().st_size <= 5000:
        raise RuntimeError(f"refinement_poster failed to produce {poster_final_png}")

    with Image.open(poster_final_png) as im:
        width, height = im.size

    elapsed = round(time.monotonic() - t0, 3)
    out_poster_dir.mkdir(parents=True, exist_ok=True)
    eval_input_png.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(poster_final_png, out_poster_png)
    shutil.copy2(poster_final_png, eval_input_png)

    stats = get_codex_run_stats()
    sha256_hex = hashlib.sha256(out_poster_png.read_bytes()).hexdigest()
    manifest_data = {
        "index": idx,
        "canonical_name": canon_name,
        "status": "completed",
        "model": model,
        "reasoning_effort": effort,
        "paperx_commit": "66bb8636a6973c7d2b2d16eddd70b76081488e33",
        "mineru_source": entry["source_auto"],
        "poster_png": str(out_poster_png),
        "poster_sha256": sha256_hex,
        "dimensions": {"width": width, "height": height},
        "wall_seconds": elapsed,
        "codex_stats": {k: v for k, v in stats.items() if k != "calls"},
        "num_sections_split": len(split_paths),
    }
    out_manifest.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    success_flag.write_text("ok\n", encoding="utf-8")
    call_cnt = stats.get("call_count", 0)
    print(
        f"✅ [{idx:03d}/100] Done in {elapsed:.1f}s | calls={call_cnt} | "
        f"size={width}x{height} | {canon_name}",
        flush=True,
    )
    return manifest_data


def _run_worker_subprocess(idx: int, args: argparse.Namespace) -> tuple[int, int, str]:
    log_dir = EXP_DIR / "logs" / "papers"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"paper_{idx:03d}.log"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--index",
        str(idx),
        "--model",
        args.model,
        "--effort",
        args.effort,
        "--timeout",
        str(args.timeout),
    ]
    if args.resume:
        cmd.append("--resume")
    with log_file.open("a", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False)
    return idx, proc.returncode, str(log_file)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, help="1-based paper index (1..100) for single-paper execution")
    ap.add_argument("--indices", type=str, help="Comma-separated 1-based indices (e.g. 1,2,3)")
    ap.add_argument("--all", action="store_true", help="Run all 100 papers")
    ap.add_argument("--jobs", type=int, default=6, help="Parallel paper workers in batch mode")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    entries = json.loads(INDEX_MAP_PATH.read_text(encoding="utf-8"))
    by_idx = {int(e["index"]): e for e in entries}

    if args.index is not None:
        if args.index not in by_idx:
            raise ValueError(f"--index must be in 1..100, got {args.index}")
        run_single_paper(
            by_idx[args.index],
            model=args.model,
            effort=args.effort,
            timeout_s=args.timeout,
            resume=args.resume,
        )
        return 0

    if args.indices:
        target_indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    elif args.all:
        target_indices = [i for i in sorted(by_idx.keys()) if i != 10] + ([10] if 10 in by_idx else [])
    else:
        ap.error("Specify --index N, --indices ..., or --all")
        return 2

    print(f"Dispatching {len(target_indices)} paper(s) with jobs={args.jobs}, model={args.model}, effort={args.effort}", flush=True)
    failures: list[tuple[int, int, str]] = []
    done_count = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(_run_worker_subprocess, idx, args): idx for idx in target_indices}
        for fut in as_completed(futs):
            idx, rc, log_path = fut.result()
            done_count += 1
            cname = by_idx[idx]["canonical_name"]
            if rc == 0:
                print(f"[{done_count}/{len(target_indices)}] OK  #{idx:03d} {cname}", flush=True)
            else:
                failures.append((idx, rc, log_path))
                print(f"[{done_count}/{len(target_indices)}] ERR #{idx:03d} rc={rc} log={log_path} ({cname})", flush=True)

    summary_path = EXP_DIR / "batch_generation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "requested": len(target_indices),
                "succeeded": len(target_indices) - len(failures),
                "failed": [{"index": i, "returncode": r, "log": lp} for i, r, lp in failures],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if failures:
        print(f"Batch finished with {len(failures)} failure(s); see {summary_path}", flush=True)
        return 1
    print(f"All {len(target_indices)} paper(s) generated successfully!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
