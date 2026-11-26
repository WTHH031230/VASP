"""Load the visual-evaluation rubrics, check model answers, and aggregate them.

The model returns a verdict and evidence for one atomic question. It does not
return a score. Pass rates are computed here from PASS and FAIL only.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DIMENSION_ORDER = (
    "aesthetic_element",
    "aesthetic_engagement",
    "aesthetic_layout",
)

VISUAL_EVALUATION_DIMENSIONS = DIMENSION_ORDER

ALLOWED_VERDICTS = frozenset(
    {"PASS", "FAIL", "UNCERTAIN", "NOT_APPLICABLE"}
)
ALLOWED_QUESTION_TYPES = frozenset({"gate", "discriminative"})

BASE_SYSTEM_PROMPT = """You are a visual verification agent for academic posters.
You are a verifier, not a holistic scorer. Evaluate every supplied atomic
question independently and only from directly observable evidence in the
attached poster image(s).

Verdict rules:
- PASS: the positive requirement is clearly satisfied by every relevant,
  visible instance. For a universal requirement, one clear counterexample is
  enough to prevent PASS.
- FAIL: there is clear visual or textual evidence that the requirement is
  violated.
- UNCERTAIN: the relevant content exists, but resolution, ambiguity, occlusion,
  or missing context prevents a reliable judgement.
- NOT_APPLICABLE: the question's explicit applicability condition is not met.
  Do not use this merely because the answer is difficult.

Evidence rules:
- Every result must cite a concrete observation.
- For FAIL, identify the exact defect and its approximate poster location.
- For PASS, cite representative positive evidence and its location.
- For UNCERTAIN or NOT_APPLICABLE, explain why that verdict is necessary.
- Do not use outside knowledge, infer invisible content, compare against an
  imagined ideal poster, or let one question's verdict determine another.
- Do not return a 1--5 rating or an overall aesthetic opinion.

Return one result for every question and output only one valid JSON object."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Rubric must be a JSON object: {path}")
    return data


def validate_rubric(rubric: Mapping[str, Any], path: Path | None = None) -> None:
    """Validate one rubric before it can be sent to a model."""

    source = str(path) if path else "<rubric>"
    dimension = rubric.get("dimension")
    if dimension not in DIMENSION_ORDER:
        raise ValueError(f"Unknown dimension {dimension!r} in {source}")
    if not isinstance(rubric.get("system_prompt"), str):
        raise ValueError(f"Missing system_prompt in {source}")

    questions = rubric.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"{dimension} must contain at least one question")

    required = {
        "id",
        "question",
        "pass_condition",
        "fail_condition",
        "applicability",
        "weight",
        "critical",
    }
    seen: set[str] = set()
    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError(f"Question {index} in {source} is not an object")
        missing = required.difference(question)
        if missing:
            raise ValueError(
                f"Question {index} in {source} is missing {sorted(missing)}"
            )
        question_id = question["id"]
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"Question {index} in {source} has an invalid id")
        if question_id in seen:
            raise ValueError(f"Duplicate question id {question_id!r} in {source}")
        seen.add(question_id)
        weight = question["weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"Question {question_id} has a non-numeric weight")
        if not math.isfinite(float(weight)) or float(weight) <= 0:
            raise ValueError(f"Question {question_id} must have a positive weight")
        if not isinstance(question["critical"], bool):
            raise ValueError(f"Question {question_id} critical must be boolean")
        question_type = question.get("question_type")
        if question_type is not None and question_type not in ALLOWED_QUESTION_TYPES:
            raise ValueError(
                f"Question {question_id} has invalid question_type "
                f"{question_type!r}; expected one of {sorted(ALLOWED_QUESTION_TYPES)}"
            )


def rubric_dir_path(rubric_dir: str | Path | None = None) -> Path:
    if rubric_dir is not None:
        return Path(rubric_dir)
    return Path(__file__).resolve().parent / "rubrics"


def load_rubrics(
    rubric_dir: str | Path | None = None,
    *,
    dimensions: Sequence[str] = DIMENSION_ORDER,
) -> dict[str, dict[str, Any]]:
    """Load and validate requested rubric files in canonical dimension order."""

    directory = rubric_dir_path(rubric_dir)
    rubrics: dict[str, dict[str, Any]] = {}
    unknown = [dimension for dimension in dimensions if dimension not in DIMENSION_ORDER]
    if unknown:
        raise ValueError(f"Unknown requested rubric dimensions: {unknown}")
    requested = set(dimensions)
    for dimension in DIMENSION_ORDER:
        if dimension not in requested:
            continue
        path = directory / f"{dimension}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing rubric: {path}")
        rubric = _read_json(path)
        validate_rubric(rubric, path)
        if rubric["dimension"] != dimension:
            raise ValueError(
                f"Rubric filename {path.name} does not match {rubric['dimension']}"
            )
        rubrics[dimension] = rubric
    return rubrics


def rubric_sha256(rubric_dir: str | Path | None = None) -> str:
    """Hash the rubric files that define the 12-question evaluation."""

    directory = rubric_dir_path(rubric_dir)
    digest = hashlib.sha256()
    for dimension in VISUAL_EVALUATION_DIMENSIONS:
        path = directory / f"{dimension}.json"
        digest.update(dimension.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_visual_evaluation(
    rubric_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble the canonical 12-question visual evaluation in rubric order."""

    rubrics = load_rubrics(
        rubric_dir,
        dimensions=VISUAL_EVALUATION_DIMENSIONS,
    )
    groups: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    for dimension in VISUAL_EVALUATION_DIMENSIONS:
        rubric = rubrics[dimension]
        group_questions: list[dict[str, Any]] = []
        for source_question in rubric["questions"]:
            question = dict(source_question)
            question["source_dimension"] = dimension
            question["source_dimension_title"] = rubric["dimension_title"]
            group_questions.append(question)
            questions.append(question)
        groups.append(
            {
                "id": dimension,
                "title": rubric["dimension_title"],
                "description": rubric["description"],
                "questions": group_questions,
            }
        )

    return {
        "system": "visual_evaluation",
        "system_title": "Visual Evaluation",
        "description": (
            "Judgement-based evaluation of an academic poster's visible "
            "figure presentation, attention guidance, and spatial layout."
        ),
        "execution_protocol": "one_independent_dialogue_per_question",
        "groups": groups,
        "questions": questions,
    }


def build_user_prompt(
    rubric: Mapping[str, Any], questions: Sequence[Mapping[str, Any]]
) -> str:
    """Build the user prompt for the supplied questions only."""

    public_questions = []
    for item in questions:
        public_question = {
            "id": item["id"],
            "question": item["question"],
        }
        if "evaluation_instructions" in item:
            public_question["evaluation_instructions"] = item[
                "evaluation_instructions"
            ]
        public_question.update(
            {
                "pass_condition": item["pass_condition"],
                "fail_condition": item["fail_condition"],
                "applicability": item["applicability"],
            }
        )
        public_questions.append(public_question)
    output_contract = {
        "dimension": rubric["dimension"],
        "results": [
            {
                "id": "question id",
                "verdict": "PASS | FAIL | UNCERTAIN | NOT_APPLICABLE",
                "evidence": "specific visible evidence or the reason evidence is unavailable",
                "location": "approximate location, e.g. top-left Methods panel; use whole poster or N/A when appropriate",
                "confidence": "number from 0.0 to 1.0",
            }
        ],
        "dimension_summary": "brief factual summary of the observed evidence",
    }
    return "\n\n".join(
        [
            f"Dimension: {rubric['dimension_title']} ({rubric['dimension']})",
            f"Scope: {rubric['description']}",
            "Evaluate the attached poster image. Answer every question below. "
            "Do not add or remove question IDs.",
            "QUESTIONS:\n"
            + json.dumps(public_questions, ensure_ascii=False, indent=2),
            "OUTPUT CONTRACT:\n"
            + json.dumps(output_contract, ensure_ascii=False, indent=2),
            "Return only the JSON object. Do not wrap it in Markdown fences.",
        ]
    )


def build_system_prompt(rubric: Mapping[str, Any]) -> str:
    """Build the verifier instructions for one dimension."""

    return (
        BASE_SYSTEM_PROMPT
        + "\n\nDIMENSION-SPECIFIC SCOPE:\n"
        + str(rubric["system_prompt"])
    )


def extract_json_object(raw_content: Any) -> dict[str, Any]:
    """Extract a JSON object from a model message."""

    if isinstance(raw_content, list):
        text_parts: list[str] = []
        for block in raw_content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
        raw_content = "".join(text_parts)
    if not isinstance(raw_content, str):
        raise ValueError("Model response content is not text")

    text = raw_content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        left, right = text.find("{"), text.rfind("}")
        if left < 0 or right <= left:
            raise ValueError("Model response contains no JSON object")
        parsed = json.loads(text[left : right + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON is not an object")
    return parsed


def validate_dimension_response(
    raw_result: Mapping[str, Any],
    rubric: Mapping[str, Any],
    questions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate one model response and attach local scoring metadata."""

    expected_questions = list(questions or rubric["questions"])
    expected_ids = [item["id"] for item in expected_questions]
    if raw_result.get("dimension") != rubric["dimension"]:
        raise ValueError(
            f"Expected dimension {rubric['dimension']!r}, got "
            f"{raw_result.get('dimension')!r}"
        )
    results = raw_result.get("results")
    if not isinstance(results, list):
        raise ValueError("Response results must be a list")

    by_id: dict[str, dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("Every result must be an object")
        item_id = item.get("id")
        if item_id in by_id:
            raise ValueError(f"Duplicate result id: {item_id}")
        if item_id not in expected_ids:
            raise ValueError(f"Unknown result id: {item_id}")
        verdict = str(item.get("verdict", "")).upper()
        if verdict not in ALLOWED_VERDICTS:
            raise ValueError(f"Invalid verdict for {item_id}: {verdict!r}")
        evidence = item.get("evidence")
        location = item.get("location")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError(f"Result {item_id} requires non-empty evidence")
        if not isinstance(location, str) or not location.strip():
            raise ValueError(f"Result {item_id} requires a non-empty location")
        confidence = item.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError(f"Result {item_id} confidence must be numeric")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Result {item_id} confidence must be within [0, 1]")
        by_id[item_id] = {
            "id": item_id,
            "verdict": verdict,
            "evidence": evidence.strip(),
            "location": location.strip(),
            "confidence": confidence,
        }

    missing = [item_id for item_id in expected_ids if item_id not in by_id]
    if missing:
        raise ValueError(f"Model omitted result IDs: {missing}")
    if len(results) != len(expected_ids):
        raise ValueError("Response result count does not match the rubric")

    metadata = {item["id"]: item for item in expected_questions}
    normalized: list[dict[str, Any]] = []
    for item_id in expected_ids:
        item = dict(by_id[item_id])
        question = metadata[item_id]
        item.update(
            {
                "dimension": rubric["dimension"],
                "question": question["question"],
                "weight": float(question["weight"]),
                "critical": bool(question["critical"]),
                "question_type": question.get("question_type", "unclassified"),
            }
        )
        normalized.append(item)

    summary = raw_result.get("dimension_summary", "")
    if not isinstance(summary, str):
        raise ValueError("dimension_summary must be a string")
    return {
        "dimension": rubric["dimension"],
        "dimension_title": rubric["dimension_title"],
        "results": normalized,
        "dimension_summary": summary.strip(),
    }


def aggregate_dimension(result: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate atomic judgements. The model does not supply a score."""

    items = result["results"]
    summary = _summarize_items(items)
    question_types = sorted(
        {str(item.get("question_type", "unclassified")) for item in items}
    )
    by_question_type = {
        question_type: _summarize_items(
            [
                item
                for item in items
                if item.get("question_type", "unclassified") == question_type
            ]
        )
        for question_type in question_types
    }

    gate_items = [
        item for item in items if item.get("question_type") == "gate"
    ]
    gate_failures = [
        item["id"] for item in gate_items if item["verdict"] == "FAIL"
    ]
    if not gate_items:
        gate_status = None
    elif gate_failures:
        gate_status = "FAIL"
    elif any(item["verdict"] == "UNCERTAIN" for item in gate_items):
        gate_status = "UNCERTAIN"
    elif any(item["verdict"] == "PASS" for item in gate_items):
        gate_status = "PASS"
    else:
        gate_status = "NOT_APPLICABLE"

    summary.update(
        {
            "by_question_type": by_question_type,
            "gate_status": gate_status,
            "gate_failures": gate_failures,
        }
    )
    return summary


def _summarize_items(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(item["verdict"] for item in items)
    evaluable = [item for item in items if item["verdict"] in {"PASS", "FAIL"}]
    total_weight = sum(float(item["weight"]) for item in evaluable)
    passed_weight = sum(
        float(item["weight"]) for item in evaluable if item["verdict"] == "PASS"
    )
    pass_rate = counts["PASS"] / len(evaluable) if evaluable else None
    weighted_pass_rate = passed_weight / total_weight if total_weight else None
    critical_failures = [
        item["id"]
        for item in items
        if item["critical"] and item["verdict"] == "FAIL"
    ]
    mean_confidence = (
        sum(float(item["confidence"]) for item in items) / len(items)
        if items
        else None
    )
    return {
        "question_count": len(items),
        "counts": {verdict: counts[verdict] for verdict in sorted(ALLOWED_VERDICTS)},
        "evaluable_count": len(evaluable),
        "coverage": len(evaluable) / len(items) if items else None,
        "pass_rate": pass_rate,
        "weighted_pass_rate": weighted_pass_rate,
        "critical_failures": critical_failures,
        "mean_confidence": mean_confidence,
    }


def aggregate_equal_groups(
    results: Mapping[str, Mapping[str, Any]],
    rubrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute the paper's equal-group Visual Presentation scores.

    Each of the three groups (Figure presentation: 3, Visual guidance and emphasis: 5,
    Layout and spatial organization: 4) receives equal 1/3 weight:
      - ``overall_fixed_denominator`` (Paper Eq. 2, main tables):
        group pass rate = PASS / |Q_g| (fixed group size; only PASS receives credit).
      - ``overall_excl_na`` (Paper Appendix C sensitivity, Table tab:na-sensitivity):
        group pass rate = PASS / |Q_g^{app}|, where |Q_g^{app}| = |Q_g| - N_NA
        (excludes NOT_APPLICABLE only; UNCERTAIN remains in the denominator with 0 credit,
        and a group with 0 applicable criteria scores 0.0).
    """
    groups_summary: dict[str, dict[str, Any]] = {}
    fixed_rates: list[float] = []
    excl_na_rates: list[float] = []

    for dimension in VISUAL_EVALUATION_DIMENSIONS:
        rubric = rubrics.get(dimension)
        if rubric is None:
            continue
        qids = [str(q["id"]) for q in rubric["questions"]]
        counts = Counter()
        for qid in qids:
            stored = results.get(qid)
            if stored is not None:
                counts[str(stored.get("verdict", ""))] += 1
        total_q = len(qids)
        applicable = total_q - counts["NOT_APPLICABLE"]
        evaluable = counts["PASS"] + counts["FAIL"]
        fixed_rate = counts["PASS"] / total_q if total_q else 0.0
        excl_na_rate = counts["PASS"] / applicable if applicable else 0.0
        fixed_rates.append(fixed_rate)
        excl_na_rates.append(excl_na_rate)
        groups_summary[dimension] = {
            "dimension_title": rubric.get("dimension_title", dimension),
            "question_count": total_q,
            "counts": {v: counts[v] for v in sorted(ALLOWED_VERDICTS)},
            "applicable_count": applicable,
            "evaluable_count": evaluable,
            "pass_rate_fixed_denominator": fixed_rate,
            "pass_rate_excl_na": excl_na_rate,
        }

    return {
        "overall_fixed_denominator": sum(fixed_rates) / len(fixed_rates) if fixed_rates else 0.0,
        "overall_excl_na": sum(excl_na_rates) / len(excl_na_rates) if excl_na_rates else 0.0,
        "groups": groups_summary,
    }


def aggregate_results(
    results: Mapping[str, Mapping[str, Any]],
    rubrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate stored answers with both flat and equal-group paper summaries."""

    items: list[Mapping[str, Any]] = []
    for dimension in VISUAL_EVALUATION_DIMENSIONS:
        rubric = rubrics.get(dimension)
        if rubric is None:
            continue
        for question in rubric["questions"]:
            stored = results.get(question["id"])
            if stored is not None:
                items.append(stored)
    return {
        "visual_evaluation": aggregate_dimension({"results": items}),
        "equal_group_summary": aggregate_equal_groups(results, rubrics),
    }


def question_ids(
    rubrics: Mapping[str, Mapping[str, Any]],
    dimensions: Iterable[str] | None = None,
) -> list[str]:
    selected = list(dimensions or VISUAL_EVALUATION_DIMENSIONS)
    return [
        question["id"]
        for dimension in selected
        for question in rubrics[dimension]["questions"]
    ]
