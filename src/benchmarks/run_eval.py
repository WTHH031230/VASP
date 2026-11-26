"""Unified 6-backend benchmark runner (codex, claude, agy, pi-api, pi-chatgpt, vertex).

Loads original prompts, rubrics, stimuli, and JSON schemas verbatim from each benchmark
module and dispatches VLM calls via `src.benchmarks.llm_backends.call_vlm_json`.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from src.benchmarks.llm_backends import BACKENDS, REPO_ROOT, call_vlm_json


def _run_vision_bench(args: argparse.Namespace) -> dict[str, Any]:
    from src.benchmarks.vision_bench.evaluator import (
        aggregate_equal_groups,
        build_system_prompt,
        build_user_prompt,
        load_rubrics,
    )
    from src.benchmarks.vision_bench.poster_stimuli import prepare_question_poster

    rubrics = load_rubrics(REPO_ROOT / "src" / "benchmarks" / "vision_bench" / "rubrics")
    schema = json.loads(
        (REPO_ROOT / "configs" / "schemas" / "vision_bench" / "single_question_response.schema.json").read_text(encoding="utf-8")
    )
    results: dict[str, list[dict[str, Any]]] = {}
    flat_answers: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="vasp_vb_stim_") as stim_tmp:
        stim_root = Path(stim_tmp)
        for dim, rubric in rubrics.items():
            dim_items: list[dict[str, Any]] = []
            for q in rubric["questions"]:
                qid = str(q["id"])
                one_rubric = {**rubric, "questions": [dict(q)]}
                # DIST01 receives the 30% scaled distance-view image centered on #7F7F7F.
                img_path, _stim_meta = prepare_question_poster(
                    poster=args.poster,
                    question=q,
                    output_dir=stim_root / qid,
                )
                prompt = f"{build_system_prompt(one_rubric)}\n\n{build_user_prompt(one_rubric, [dict(q)])}"
                parsed, meta = call_vlm_json(
                    backend=args.backend,
                    prompt=prompt,
                    images=[img_path],
                    schema=schema,
                    model=args.model,
                    effort=args.effort,
                    provider=args.provider,
                    timeout=args.timeout,
                )
                dim_items.append({"question_id": qid, "response": parsed, "meta": meta})
                if isinstance(parsed, dict):
                    rlist = parsed.get("results")
                    if isinstance(rlist, list) and rlist and isinstance(rlist[0], dict):
                        flat_answers[qid] = rlist[0]
                    elif "verdict" in parsed:
                        flat_answers[qid] = parsed
            results[dim] = dim_items
    summary = aggregate_equal_groups(flat_answers, rubrics)
    return {
        "benchmark": "vision_bench",
        "poster": str(args.poster),
        "backend": args.backend,
        "model": args.model,
        "summary": summary,
        "results": results,
    }


def _run_info_coverage(args: argparse.Namespace) -> dict[str, Any]:
    if not args.reference_units:
        raise SystemExit("--reference-units is required for information_coverage")
    prompt_tpl = (REPO_ROOT / "src" / "benchmarks" / "information_coverage" / "evaluation_prompt.md").read_text(encoding="utf-8")
    schema = json.loads(
        (REPO_ROOT / "configs" / "schemas" / "information_coverage" / "evaluation_output_schema.json").read_text(encoding="utf-8")
    )
    ref_data = json.loads(args.reference_units.read_text(encoding="utf-8"))
    units = ref_data
    if isinstance(ref_data, dict) and isinstance(ref_data.get("posters"), list):
        posters = ref_data["posters"]
        if args.paper_id:
            key_norm = args.paper_id.strip().lower()
            matched = [
                item for item in posters
                if isinstance(item, dict) and any(
                    str(item.get(k) or "").strip().lower() == key_norm
                    for k in ("poster_id", "paper", "paper_id", "paper_name", "title")
                )
            ]
            if not matched:
                tokens = set(key_norm.split())
                scored = sorted(
                    posters,
                    key=lambda it: len(tokens & set(str(it.get("title") or it.get("poster_id") or "").lower().split())),
                    reverse=True,
                )
                if scored and len(tokens & set(str(scored[0].get("title") or "").lower().split())) >= max(2, len(tokens) - 2):
                    matched = [scored[0]]
            if not matched:
                raise SystemExit(f"paper_id {args.paper_id!r} not found in {args.reference_units}")
            units = matched[0]
        else:
            units = posters[0]
    elif args.paper_id and isinstance(ref_data, dict):
        units = ref_data.get(args.paper_id, ref_data)
    prompt = f"{prompt_tpl}\n\nReference Information Units (JSON):\n{json.dumps(units, ensure_ascii=False, indent=2)}"
    parsed, meta = call_vlm_json(
        backend=args.backend,
        prompt=prompt,
        images=[args.poster],
        schema=schema,
        model=args.model,
        effort=args.effort,
        provider=args.provider,
        timeout=args.timeout,
    )
    return {"benchmark": "information_coverage", "poster": str(args.poster), "backend": args.backend, "model": args.model, "result": parsed, "meta": meta}


def _run_vlm_judge(args: argparse.Namespace) -> dict[str, Any]:
    from src.benchmarks.vlm_judge import rubrics, schemas

    out: dict[str, Any] = {}
    for aspect in rubrics.ALL_ASPECTS:
        prompt_text = rubrics.independent_prompt(aspect, image_count=1)
        schema = schemas.single_aspect_schema()
        parsed, meta = call_vlm_json(
            backend=args.backend,
            prompt=prompt_text,
            images=[args.poster],
            schema=schema,
            model=args.model,
            effort=args.effort,
            provider=args.provider,
            timeout=args.timeout,
        )
        out[aspect] = {"response": parsed, "meta": meta}
    return {"benchmark": "vlm_judge", "poster": str(args.poster), "backend": args.backend, "model": args.model, "results": out}


def _load_upstream_paperquiz_template() -> tuple[str, str]:
    tpl_path = REPO_ROOT / "src" / "benchmarks" / "paperquiz" / "upstream_templates" / "answer_question_from_image.yaml"
    import yaml
    cfg = yaml.safe_load(tpl_path.read_text(encoding="utf-8"))
    return str(cfg["system_prompt"]), str(cfg["template"])


def _run_paperquiz(args: argparse.Namespace) -> dict[str, Any]:
    if not args.qa_json:
        raise SystemExit("--qa-json is required for paperquiz")
    qa_data = json.loads(args.qa_json.read_text(encoding="utf-8"))
    sys_prompt, user_tpl = _load_upstream_paperquiz_template()
    from jinja2 import Environment, StrictUndefined
    tpl = Environment(undefined=StrictUndefined).from_string(user_tpl)

    splits: list[tuple[str, dict[str, Any]]] = []
    if isinstance(qa_data, dict) and ("detail" in qa_data or "understanding" in qa_data):
        for split in ("detail", "understanding"):
            blk = qa_data.get(split)
            if isinstance(blk, dict) and blk.get("questions"):
                qs = blk["questions"]
                q_map = qs if isinstance(qs, dict) else {f"Question {i+1}": q for i, q in enumerate(qs)}
                splits.append((split, q_map))
    elif isinstance(qa_data, dict):
        splits.append(("all", qa_data.get("questions", qa_data)))
    elif isinstance(qa_data, list):
        splits.append(("all", {f"Question {i+1}": q for i, q in enumerate(qa_data)}))

    results: dict[str, Any] = {}
    for split_name, q_map in splits:
        rendered_user = tpl.render(questions=q_map)
        prompt = f"{sys_prompt}\n\n{rendered_user}"
        schema = {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "reference": {"type": "string"},
                },
                "required": ["answer"],
            },
        }
        parsed, meta = call_vlm_json(
            backend=args.backend,
            prompt=prompt,
            images=[args.poster],
            schema=schema,
            model=args.model,
            effort=args.effort,
            provider=args.provider,
            timeout=args.timeout,
        )
        results[split_name] = {"answers": parsed, "meta": meta}
    return {"benchmark": "paperquiz", "poster": str(args.poster), "backend": args.backend, "model": args.model, "results": results}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run VASP VLM benchmarks across 6 supported backends (including Vertex AI API).")
    ap.add_argument("benchmark", choices=("vision_bench", "information_coverage", "vlm_judge", "paperquiz"))
    ap.add_argument("--poster", type=Path, required=True, help="Path to rendered poster PNG.")
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    ap.add_argument("--backend", choices=BACKENDS, default="codex")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--provider", default="openai-compat")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--reference-units", type=Path, help="Reference units JSON for information_coverage.")
    ap.add_argument("--paper-id", default="", help="Optional paper key inside reference-units JSON.")
    ap.add_argument("--qa-json", type=Path, help="Per-paper PaperQuiz o3_qa.json.")
    args = ap.parse_args()

    dispatch = {
        "vision_bench": _run_vision_bench,
        "information_coverage": _run_info_coverage,
        "vlm_judge": _run_vlm_judge,
        "paperquiz": _run_paperquiz,
    }
    report = dispatch[args.benchmark](args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.benchmark} report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
