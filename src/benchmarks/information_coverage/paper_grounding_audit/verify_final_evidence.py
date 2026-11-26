from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

if __package__ in {None, ""}:
    import adjudicate_flags as adjudication
else:
    from . import adjudicate_flags as adjudication
if __package__ in {None, ""}:
    import run_audit as base
else:
    from . import run_audit as base


ROOT = Path(__file__).resolve().parent / "paper_grounding_audit"
OUTPUT = ROOT / "evidence_verification.json"


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in value if character.isalnum())


def matches_page(quote: str, page_text: str) -> tuple[bool, str]:
    normalized_page = normalize(page_text)
    normalized_quote = normalize(quote)
    if normalized_quote and normalized_quote in normalized_page:
        return True, "exact_normalized"
    segments = [
        normalize(segment)
        for segment in re.split(r"(?:\.{3,}|…|\[\.\.\.\])", quote)
        if len(normalize(segment)) >= 8
    ]
    if len(segments) >= 2 and all(segment in normalized_page for segment in segments):
        return True, "ellipsis_segments"
    return False, "unmatched"


def main() -> None:
    references, _ = base.load_inputs()
    checked = 0
    verified = 0
    match_modes = {"exact_normalized": 0, "ellipsis_segments": 0}
    missing_evidence_units = []
    unmatched = []
    invalid_pages = []
    for reference in references:
        poster_id = reference["poster_id"]
        initial = base.read_json(base.RESULT_ROOT / f"{poster_id}.json")
        flags = adjudication.flagged_units(initial)
        if not flags:
            continue
        result = base.read_json(adjudication.RESULT_ROOT / f"{poster_id}.json")
        for review in result["reviews"]:
            if review["final_verdict"] == "SUPPORTED":
                continue
            evidence = review["evidence"]
            if not evidence and review["final_verdict"] != "NOT_FOUND":
                missing_evidence_units.append(review["unit_id"])
            for item in evidence:
                checked += 1
                page = item["page"]
                page_path = (
                    ROOT / "paper_pages" / poster_id / f"page-{page:03d}.txt"
                )
                if not page_path.is_file():
                    invalid_pages.append(
                        {"unit_id": review["unit_id"], "page": page}
                    )
                    continue
                page_text = page_path.read_text(encoding="utf-8", errors="replace")
                matched, mode = matches_page(item["quote"], page_text)
                if matched:
                    verified += 1
                    match_modes[mode] += 1
                else:
                    unmatched.append(
                        {
                            "unit_id": review["unit_id"],
                            "page": page,
                            "location": item["location"],
                            "quote": item["quote"],
                        }
                    )
    report = {
        "evidence_excerpt_count": checked,
        "verified_page_matches": verified,
        "match_modes": match_modes,
        "unmatched_excerpt_count": len(unmatched),
        "invalid_page_count": len(invalid_pages),
        "non_not_found_units_missing_evidence": missing_evidence_units,
        "invalid_pages": invalid_pages,
        "unmatched_excerpts": unmatched,
        "normalization": "Unicode NFKC + casefold + alphanumeric characters only",
    }
    base.write_json(OUTPUT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
