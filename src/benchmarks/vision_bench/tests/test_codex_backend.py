from __future__ import annotations
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace

from eval_poster import (
    aggregate_results,
    build_codex_command,
    ensure_resume_compatible,
    existing_report_action,
    make_codex_cwd,
    questions_to_run,
    targeted_prompt,
)
from evaluator import VISUAL_EVALUATION_DIMENSIONS, load_rubrics


class CodexBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.rubrics = load_rubrics(dimensions=VISUAL_EVALUATION_DIMENSIONS)

    def test_targeted_prompt_contains_only_the_requested_question(self) -> None:
        rubric = self.rubrics["aesthetic_element"]
        question = next(item for item in rubric["questions"] if item["id"] == "AE13")
        prompt = targeted_prompt(rubric, question)
        self.assertIn("AE13", prompt)
        self.assertNotIn("AE04", prompt)
        self.assertNotIn("IS01", prompt)
        self.assertIn("Do not inspect files", prompt)
        self.assertNotIn("question_type", prompt)
        self.assertNotIn("discriminative question", prompt.lower())

    def test_command_is_ephemeral_read_only_and_stdin_driven(self) -> None:
        fixture = self.root / "tests" / "fixture.svg"
        schema = self.root.parents[2] / "configs" / "schemas" / "vision_bench" / "single_question_response.schema.json"
        args = SimpleNamespace(
            codex_cwd=str(self.root),
            output_schema=str(schema),
            poster=[str(fixture)],
            ignore_user_config=False,
            profile=None,
            model="test-model",
            reasoning_effort="high",
        )
        command = build_codex_command(
            args,
            codex_binary="codex",
            last_message_path=self.root / "tests" / "unused-result.json",
        )
        self.assertEqual(command[:2], ["codex", "exec"])
        self.assertIn("--ephemeral", command)
        self.assertIn("read-only", command)
        self.assertIn("--output-schema", command)
        self.assertIn("--output-last-message", command)
        self.assertIn("--image", command)
        self.assertNotIn("resume", command)
        self.assertIn("--model", command)
        self.assertIn("test-model", command)

    def test_command_omits_unset_model_and_reasoning(self) -> None:
        fixture = self.root / "tests" / "fixture.svg"
        schema = self.root.parents[2] / "configs" / "schemas" / "vision_bench" / "single_question_response.schema.json"
        args = SimpleNamespace(
            codex_cwd=str(self.root),
            output_schema=str(schema),
            poster=[str(fixture)],
            ignore_user_config=False,
            profile=None,
            model=None,
            reasoning_effort=None,
        )
        command = build_codex_command(
            args,
            codex_binary="codex",
            last_message_path=self.root / "tests" / "unused-result.json",
        )
        self.assertNotIn("--model", command)
        self.assertNotIn("--config", command)

    def test_default_codex_cwd_is_an_empty_directory(self) -> None:
        cwd, owned = make_codex_cwd(None)
        self.addCleanup(lambda: shutil.rmtree(cwd, ignore_errors=True))
        self.assertTrue(owned)
        self.assertEqual(list(cwd.iterdir()), [])
        self.assertNotEqual(cwd, self.root)

    def test_resume_keeps_answers_outside_the_requested_subset(self) -> None:
        stored = {
            "AE04": {
                "id": "AE04",
                "verdict": "PASS",
                "weight": 1.0,
                "critical": True,
                "confidence": 1.0,
                "question_type": "gate",
            },
            "IS01": {
                "id": "IS01",
                "verdict": "FAIL",
                "weight": 1.5,
                "critical": False,
                "confidence": 1.0,
                "question_type": "discriminative",
            },
        }
        self.assertEqual(questions_to_run(["AE13"], stored), ["AE13"])
        merged = dict(stored)
        merged["AE13"] = {
            "id": "AE13",
            "verdict": "PASS",
            "weight": 1.5,
            "critical": False,
            "confidence": 1.0,
            "question_type": "discriminative",
        }
        summary = aggregate_results(merged, self.rubrics)
        counted = summary["visual_evaluation"]["question_count"]
        self.assertEqual(counted, 3)
        self.assertEqual(set(merged), {"AE04", "IS01", "AE13"})

    def test_existing_answers_are_kept_unless_resume_is_set(self) -> None:
        answered = {"results": {"AE04": {"verdict": "PASS"}}, "errors": {}}
        self.assertEqual(existing_report_action(answered, resume=False), "skip")
        self.assertEqual(existing_report_action(answered, resume=True), "resume")
        dry_run = {"results": {}, "errors": {}, "status": "dry_run"}
        self.assertEqual(existing_report_action(dry_run, resume=False), "replace")

    def test_resume_rejects_a_different_rubric_hash(self) -> None:
        report = {"rubric_sha256": "abc", "model": None, "reasoning_effort": None}
        with self.assertRaises(ValueError):
            ensure_resume_compatible(
                report,
                fingerprint="def",
                model=None,
                reasoning_effort=None,
            )


if __name__ == "__main__":
    unittest.main()
