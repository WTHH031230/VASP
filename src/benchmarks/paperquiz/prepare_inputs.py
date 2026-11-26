#!/usr/bin/env python3
"""Freeze the EXP-024 inputs: posters of both groups, the upstream quiz, the upstream template.

Copies (never links) every file into the experiment directory and records source
paths and sha256 in inputs/manifest.json and _code/CODE.sha256 sources.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
import os
from pathlib import Path

ROOT = Path(os.environ.get("VASP_REPO_ROOT", str(Path(__file__).resolve().parents[3])))
UPSTREAM = ROOT / "third_party/Paper2Poster"
EXP = ROOT / "results/exp/paperquiz-upstream-api-pi-rs-20260926"
SOURCES = {
    "pi_full": lambda paper: ROOT / "results/exp/pi-full100-sol-20260925/runs/g1" / paper / "poster.png",
    "researchstudio": lambda paper: Path(
        os.environ.get("VASP_POSTERS_DIR", "./posters")
    ) / f"{paper}.png",
}
UPSTREAM_FILES = [
    "utils/prompt_templates/answer_question_from_image.yaml",
    "utils/poster_eval_utils.py",
    "utils/src/utils.py",
    "Paper2Poster-eval/eval_poster_pipeline.py",
    "camel/messages/base.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    papers = sorted(p.name for p in (ROOT / "dataset").iterdir() if p.is_dir())
    assert len(papers) == 100, len(papers)
    inputs = EXP / "inputs"
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "papers": papers,
        "groups": {},
        "quiz": {},
        "upstream": {"root": str(UPSTREAM), "commit": "623d042", "files": {}},
    }
    for group, source_of in SOURCES.items():
        entries = {}
        for paper in papers:
            source = source_of(paper)
            target = inputs / group / f"{paper}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            digest = sha256(source)
            assert sha256(target) == digest
            entries[paper] = {"source": str(source), "sha256": digest}
        manifest["groups"][group] = entries
    for paper in papers:
        source = ROOT / "dataset" / paper / "o3_qa.json"
        target = inputs / "quiz" / paper / "o3_qa.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        quiz = json.loads(target.read_text(encoding="utf-8"))
        counts = {q: len(quiz[q]["questions"]) for q in ("detail", "understanding")}
        assert counts == {"detail": 50, "understanding": 50}, (paper, counts)
        for q in ("detail", "understanding"):
            assert set(quiz[q]["questions"]) == set(quiz[q]["answers"]) == set(quiz[q]["aspects"]), (paper, q)
        manifest["quiz"][paper] = {"source": str(source), "sha256": sha256(source), "counts": counts}
    template_dir = EXP / "_code/upstream_templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(UPSTREAM / UPSTREAM_FILES[0], template_dir / "answer_question_from_image.yaml")
    for rel in UPSTREAM_FILES:
        manifest["upstream"]["files"][rel] = sha256(UPSTREAM / rel)
    assert sha256(template_dir / "answer_question_from_image.yaml") == manifest["upstream"]["files"][UPSTREAM_FILES[0]]
    (inputs / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "papers": len(papers),
        "groups": {g: len(e) for g, e in manifest["groups"].items()},
        "quiz": len(manifest["quiz"]),
        "upstream": manifest["upstream"]["files"],
    }, indent=2))


if __name__ == "__main__":
    main()
