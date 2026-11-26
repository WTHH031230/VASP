from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from eval_poster import select_questions
from evaluator import (
    VISUAL_EVALUATION_DIMENSIONS,
    aggregate_dimension,
    aggregate_equal_groups,
    load_rubrics,
    load_visual_evaluation,
    rubric_sha256,
)
from poster_stimuli import prepare_question_poster


EXPECTED_IDS = [
    "AE04",
    "AE13",
    "IS01",
    "ENH01",
    "DIST01",
    "COL01",
    "EN05",
    "EN07",
    "AL01",
    "AL05",
    "DEN01",
    "WS01",
]


class VisualRubricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rubrics = load_rubrics(dimensions=VISUAL_EVALUATION_DIMENSIONS)

    def test_default_loader_reads_only_the_three_visual_rubrics(self) -> None:
        rubrics = load_rubrics()
        self.assertEqual(list(rubrics), list(VISUAL_EVALUATION_DIMENSIONS))
        self.assertEqual(len(rubric_sha256()), 64)
        self.assertEqual(rubric_sha256(), rubric_sha256())

    def test_visual_system_has_exactly_the_twelve_current_questions(self) -> None:
        system = load_visual_evaluation()
        self.assertEqual(system["system"], "visual_evaluation")
        self.assertEqual(
            [question["id"] for question in system["questions"]],
            EXPECTED_IDS,
        )

    def test_gate_and_discriminative_counts(self) -> None:
        expected = {
            "aesthetic_element": {"gate": 1, "discriminative": 2},
            "aesthetic_engagement": {"gate": 1, "discriminative": 4},
            "aesthetic_layout": {"gate": 3, "discriminative": 1},
        }
        for dimension, counts in expected.items():
            actual = {"gate": 0, "discriminative": 0}
            for question in self.rubrics[dimension]["questions"]:
                actual[question["question_type"]] += 1
            self.assertEqual(actual, counts)

    def test_select_questions_keeps_only_requested_atomic_questions(self) -> None:
        selected, dimensions = select_questions(
            self.rubrics,
            list(VISUAL_EVALUATION_DIMENSIONS),
            ["AE13", "EN07"],
        )
        selected_ids = [
            question["id"]
            for dimension in dimensions
            for question in selected[dimension]["questions"]
        ]
        self.assertEqual(selected_ids, ["AE13", "EN07"])

    def test_aggregation_excludes_uncertain_and_not_applicable(self) -> None:
        block = {
            "results": [
                {
                    "id": "A",
                    "verdict": "PASS",
                    "weight": 2.0,
                    "critical": False,
                    "confidence": 1.0,
                    "question_type": "discriminative",
                },
                {
                    "id": "B",
                    "verdict": "FAIL",
                    "weight": 1.0,
                    "critical": True,
                    "confidence": 1.0,
                    "question_type": "gate",
                },
                {
                    "id": "C",
                    "verdict": "UNCERTAIN",
                    "weight": 10.0,
                    "critical": False,
                    "confidence": 0.5,
                    "question_type": "discriminative",
                },
                {
                    "id": "D",
                    "verdict": "NOT_APPLICABLE",
                    "weight": 10.0,
                    "critical": False,
                    "confidence": 0.5,
                    "question_type": "gate",
                },
            ]
        }
        aggregate = aggregate_dimension(block)
        self.assertEqual(aggregate["evaluable_count"], 2)
        self.assertAlmostEqual(aggregate["pass_rate"], 0.5)
        self.assertAlmostEqual(aggregate["weighted_pass_rate"], 2.0 / 3.0)
        self.assertEqual(aggregate["critical_failures"], ["B"])
        self.assertEqual(aggregate["gate_status"], "FAIL")

    def test_equal_group_aggregation_fixed_and_excl_na(self) -> None:
        # Group 1 (3 qs): all NOT_APPLICABLE -> fixed=0.0, excl_na=0.0
        # Group 2 (5 qs): 4 PASS, 1 NOT_APPLICABLE -> fixed=4/5=0.8, excl_na=4/4=1.0
        # Group 3 (4 qs): 3 PASS, 1 FAIL -> fixed=3/4=0.75, excl_na=3/4=0.75
        answers = {
            "AE04": {"id": "AE04", "verdict": "NOT_APPLICABLE"},
            "AE13": {"id": "AE13", "verdict": "NOT_APPLICABLE"},
            "IS01": {"id": "IS01", "verdict": "NOT_APPLICABLE"},
            "ENH01": {"id": "ENH01", "verdict": "PASS"},
            "DIST01": {"id": "DIST01", "verdict": "PASS"},
            "COL01": {"id": "COL01", "verdict": "PASS"},
            "EN05": {"id": "EN05", "verdict": "NOT_APPLICABLE"},
            "EN07": {"id": "EN07", "verdict": "PASS"},
            "AL01": {"id": "AL01", "verdict": "PASS"},
            "AL05": {"id": "AL05", "verdict": "PASS"},
            "DEN01": {"id": "DEN01", "verdict": "PASS"},
            "WS01": {"id": "WS01", "verdict": "FAIL"},
        }
        eq = aggregate_equal_groups(answers, self.rubrics)
        self.assertAlmostEqual(eq["groups"]["aesthetic_element"]["pass_rate_fixed_denominator"], 0.0)
        self.assertAlmostEqual(eq["groups"]["aesthetic_element"]["pass_rate_excl_na"], 0.0)
        self.assertAlmostEqual(eq["groups"]["aesthetic_engagement"]["pass_rate_fixed_denominator"], 0.8)
        self.assertAlmostEqual(eq["groups"]["aesthetic_engagement"]["pass_rate_excl_na"], 1.0)
        self.assertAlmostEqual(eq["groups"]["aesthetic_layout"]["pass_rate_fixed_denominator"], 0.75)
        self.assertAlmostEqual(eq["groups"]["aesthetic_layout"]["pass_rate_excl_na"], 0.75)
        self.assertAlmostEqual(eq["overall_fixed_denominator"], (0.0 + 0.8 + 0.75) / 3.0)
        self.assertAlmostEqual(eq["overall_excl_na"], (0.0 + 1.0 + 0.75) / 3.0)

    def test_equal_group_excl_na_retains_uncertain_in_denominator(self) -> None:
        # Group 1 (3 qs): 2 PASS, 1 UNCERTAIN -> fixed=2/3, excl_na=2/3 (NOT 1.0)
        # Group 2 (5 qs): 3 PASS, 1 UNCERTAIN, 1 NOT_APPLICABLE -> fixed=3/5, excl_na=3/4
        # Group 3 (4 qs): 4 NOT_APPLICABLE -> fixed=0.0, excl_na=0.0
        answers = {
            "AE04": {"id": "AE04", "verdict": "PASS"},
            "AE13": {"id": "AE13", "verdict": "PASS"},
            "IS01": {"id": "IS01", "verdict": "UNCERTAIN"},
            "ENH01": {"id": "ENH01", "verdict": "PASS"},
            "DIST01": {"id": "DIST01", "verdict": "PASS"},
            "COL01": {"id": "COL01", "verdict": "PASS"},
            "EN05": {"id": "EN05", "verdict": "NOT_APPLICABLE"},
            "EN07": {"id": "EN07", "verdict": "UNCERTAIN"},
            "AL01": {"id": "AL01", "verdict": "NOT_APPLICABLE"},
            "AL05": {"id": "AL05", "verdict": "NOT_APPLICABLE"},
            "DEN01": {"id": "DEN01", "verdict": "NOT_APPLICABLE"},
            "WS01": {"id": "WS01", "verdict": "NOT_APPLICABLE"},
        }
        eq = aggregate_equal_groups(answers, self.rubrics)
        self.assertEqual(eq["groups"]["aesthetic_element"]["applicable_count"], 3)
        self.assertAlmostEqual(eq["groups"]["aesthetic_element"]["pass_rate_fixed_denominator"], 2.0 / 3.0)
        self.assertAlmostEqual(eq["groups"]["aesthetic_element"]["pass_rate_excl_na"], 2.0 / 3.0)
        self.assertEqual(eq["groups"]["aesthetic_engagement"]["applicable_count"], 4)
        self.assertAlmostEqual(eq["groups"]["aesthetic_engagement"]["pass_rate_fixed_denominator"], 0.6)
        self.assertAlmostEqual(eq["groups"]["aesthetic_engagement"]["pass_rate_excl_na"], 0.75)
        self.assertEqual(eq["groups"]["aesthetic_layout"]["applicable_count"], 0)
        self.assertAlmostEqual(eq["groups"]["aesthetic_layout"]["pass_rate_fixed_denominator"], 0.0)
        self.assertAlmostEqual(eq["groups"]["aesthetic_layout"]["pass_rate_excl_na"], 0.0)
        self.assertAlmostEqual(eq["overall_fixed_denominator"], (2.0 / 3.0 + 0.6 + 0.0) / 3.0)
        self.assertAlmostEqual(eq["overall_excl_na"], (2.0 / 3.0 + 0.75 + 0.0) / 3.0)

    def test_normalize_report_populates_dimension_aggregates(self) -> None:
        from eval_poster import normalize_report
        from evaluator import validate_dimension_response

        report: dict[str, object] = {"dimensions": {}}
        dims: dict[str, object] = {}
        for dim_name, rubric in self.rubrics.items():
            validated = validate_dimension_response(
                {
                    "dimension": dim_name,
                    "results": [
                        {
                            "id": q["id"],
                            "verdict": "PASS" if idx % 2 == 0 else "FAIL",
                            "evidence": "Observed on poster.",
                            "location": "center",
                            "confidence": 0.9,
                        }
                        for idx, q in enumerate(rubric["questions"])
                    ],
                    "dimension_summary": "Summary.",
                },
                rubric,
            )
            dims[dim_name] = validated
        report["dimensions"] = dims
        normalize_report(report, self.rubrics, list(self.rubrics.keys()))
        for dim_name in self.rubrics:
            agg = report["dimensions"][dim_name]["aggregate"]
            self.assertIsNotNone(agg["pass_rate"])
        self.assertIn("equal_group_summary", report["aggregate"])
        self.assertIn("equal_group_summary", report["summary"])

    def test_refresh_report_populates_dimensions_and_aggregates(self) -> None:
        from eval_poster import refresh_report
        from evaluator import validate_dimension_response

        flat_results: dict[str, object] = {}
        for dim_name, rubric in self.rubrics.items():
            validated = validate_dimension_response(
                {
                    "dimension": dim_name,
                    "results": [
                        {
                            "id": q["id"],
                            "verdict": "PASS",
                            "evidence": "Observed on poster.",
                            "location": "center",
                            "confidence": 0.9,
                        }
                        for q in rubric["questions"]
                    ],
                    "dimension_summary": "Summary.",
                },
                rubric,
            )
            for item in validated["results"]:
                flat_results[item["id"]] = item
        report: dict[str, object] = {"results": flat_results}
        refresh_report(report, self.rubrics, status="completed")
        for dim_name, rubric in self.rubrics.items():
            dim_block = report["dimensions"][dim_name]
            self.assertEqual(len(dim_block["results"]), len(rubric["questions"]))
            self.assertEqual(dim_block["aggregate"]["question_count"], len(rubric["questions"]))
            self.assertAlmostEqual(dim_block["aggregate"]["pass_rate"], 1.0)
        self.assertIn("equal_group_summary", report["aggregate"])

    def test_dist01_uses_centered_thirty_percent_stimulus(self) -> None:
        question = next(
            item
            for item in self.rubrics["aesthetic_engagement"]["questions"]
            if item["id"] == "DIST01"
        )
        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            source = directory / "poster.png"
            Image.new("RGB", (200, 100), "white").save(source)
            output, metadata = prepare_question_poster(
                source, question, directory / "stimulus"
            )
            self.assertIsNotNone(metadata)
            assert metadata is not None
            source_width, source_height = metadata["source_size"]
            embedded_width, embedded_height = metadata["embedded_poster_size"]
            offset_x, offset_y = metadata["embedded_poster_offset"]
            self.assertTrue(output.is_file())
            with Image.open(output) as image:
                self.assertEqual(image.size, (source_width, source_height))
                self.assertEqual(image.getpixel((0, 0)), (127, 127, 127))
                self.assertEqual(image.getpixel((100, 50)), (255, 255, 255))
            self.assertEqual(metadata["linear_scale"], 0.3)
            self.assertEqual(metadata["canvas_background"], "#7F7F7F")
            self.assertEqual(embedded_width, round(source_width * 0.3))
            self.assertEqual(embedded_height, round(source_height * 0.3))
            self.assertEqual(offset_x, (source_width - embedded_width) // 2)
            self.assertEqual(offset_y, (source_height - embedded_height) // 2)


if __name__ == "__main__":
    unittest.main()
