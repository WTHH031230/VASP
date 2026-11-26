"""CLI for API-style Paper2Poster VLM judging through Codex or Claude."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .batch import run_batch
from .codex_exec import DEFAULT_LEDGER
from .judge import JudgeConfig, JudgeError, judge_poster


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        default="codex",
        choices=["codex", "claude-cli"],
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["low", "medium", "high", "xhigh", "max", "ultra"],
    )
    parser.add_argument(
        "--call-mode",
        default="independent",
        choices=["independent", "combined"],
        help="independent matches upstream Paper2Poster; combined uses one cheaper call",
    )
    parser.add_argument(
        "--scope",
        default="all",
        choices=["all", "aesthetic", "information"],
    )
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--aspect-jobs",
        type=int,
        default=1,
        help="parallel independent rubric calls per poster (maximum: 6)",
    )
    parser.add_argument(
        "--usage-ledger",
        type=Path,
        default=DEFAULT_LEDGER,
        help="global JSONL usage ledger",
    )
    parser.add_argument(
        "--no-global-ledger",
        action="store_true",
        help="do not append call usage to the global ledger",
    )
    parser.add_argument(
        "--codex-extra-arg",
        action="append",
        default=[],
        help="additional raw codex exec argument; repeat as needed",
    )
    parser.add_argument(
        "--claude-extra-arg",
        action="append",
        default=[],
        help="additional raw claude argument; repeat as needed",
    )


def _config(args: argparse.Namespace) -> JudgeConfig:
    return JudgeConfig(
        backend=args.backend,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        call_mode=args.call_mode,
        scope=args.scope,
        codex_bin=args.codex_bin,
        claude_bin=args.claude_bin,
        timeout_s=args.timeout,
        retries=args.retries,
        aspect_jobs=args.aspect_jobs,
        ledger_path=None if args.no_global_ledger else args.usage_ledger,
        extra_codex_args=tuple(args.codex_extra_arg),
        extra_claude_args=tuple(args.claude_extra_arg),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.benchmarks.vlm_judge",
        description=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    judge = subparsers.add_parser("judge", help="judge one poster")
    judge.add_argument("images", nargs="+", type=Path)
    judge.add_argument("--paper-id", default=None)
    judge.add_argument("--out-dir", type=Path, required=True)
    judge.add_argument("--check", action="store_true")
    _add_runtime_arguments(judge)

    batch = subparsers.add_parser("batch", help="judge a directory of posters")
    batch.add_argument("input_root", type=Path)
    batch.add_argument("output_root", type=Path)
    batch.add_argument("--poster-name", default="poster.png")
    batch.add_argument(
        "--expected-paper-root",
        type=Path,
        default=None,
        help="dataset root defining the complete expected paper set",
    )
    batch.add_argument("--jobs", type=int, default=4)
    batch.add_argument("--limit", type=int, default=None)
    batch.add_argument("--force", action="store_true")
    batch.add_argument("--check", action="store_true")
    _add_runtime_arguments(batch)
    return parser


def _preflight(args: argparse.Namespace) -> int:
    binary = args.codex_bin if args.backend == "codex" else args.claude_bin
    executable = shutil.which(binary)
    if executable is None:
        print(
            f"error: cannot find {args.backend} executable {binary!r}",
            file=sys.stderr,
        )
        return 2
    if args.backend == "claude-cli" and args.reasoning_effort == "ultra":
        print("error: Claude CLI does not support ultra effort", file=sys.stderr)
        return 2
    if args.command == "judge":
        missing = [str(path) for path in args.images if not path.expanduser().is_file()]
        if missing:
            print(f"error: missing images: {missing}", file=sys.stderr)
            return 2
    else:
        if not args.input_root.expanduser().is_dir():
            print(f"error: input root is not a directory: {args.input_root}", file=sys.stderr)
            return 2
        if (
            args.expected_paper_root is not None
            and not args.expected_paper_root.expanduser().is_dir()
        ):
            print(
                f"error: expected paper root is not a directory: {args.expected_paper_root}",
                file=sys.stderr,
            )
            return 2
    print(
        json.dumps(
            {
                "ok": True,
                "backend": args.backend,
                "executable": executable,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "call_mode": args.call_mode,
                "scope": args.scope,
                "aspect_jobs": args.aspect_jobs,
                "tool_policy": "disabled-fail-closed",
                "global_ledger": None
                if args.no_global_ledger
                else str(args.usage_ledger.expanduser()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.check:
        return _preflight(args)
    if args.backend == "claude-cli" and args.reasoning_effort == "ultra":
        print("error: Claude CLI does not support ultra effort", file=sys.stderr)
        return 2
    config = _config(args)
    if args.command == "judge":
        paper_id = args.paper_id or args.images[0].parent.name
        try:
            result = judge_poster(
                args.images,
                paper_id=paper_id,
                output_dir=args.out_dir,
                config=config,
            )
        except JudgeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "paper_id": result["paper_id"],
                    "overall_average": result["overall_average"],
                    "generation_seconds": result["generation_seconds"],
                    "calls": result["judge_usage"]["call_count"],
                    "estimated_api_cost_usd": result["judge_usage"][
                        "estimated_api_cost_usd"
                    ],
                    "result": str(args.out_dir.expanduser().resolve() / "judge_result.json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    try:
        summary = run_batch(
            args.input_root,
            args.output_root,
            config=config,
            poster_name=args.poster_name,
            expected_paper_root=args.expected_paper_root,
            jobs=args.jobs,
            limit=args.limit,
            force=args.force,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
