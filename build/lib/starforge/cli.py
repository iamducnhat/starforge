from __future__ import annotations

import argparse
import json
from typing import Any

from . import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="starforge", description="General-purpose autonomous agent runtime")
    subparsers = parser.add_subparsers(dest="subcommand")

    run_parser = subparsers.add_parser("run", help="Run an objective")
    run_parser.add_argument("objective", nargs="?", default="")
    run_parser.add_argument("--objective", dest="objective_flag", default="")
    run_parser.add_argument("--working-dir", default=".")
    run_parser.add_argument("--constraint", action="append", default=[])
    run_parser.add_argument("--max-steps", type=int, default=8)
    run_parser.add_argument("--mode", default="autonomous")
    run_parser.add_argument("--adapter", choices=["cli", "code", "api"], default="cli")
    run_parser.add_argument("--command", dest="commands", action="append", default=[])
    run_parser.add_argument("--api-url", action="append", default=[])
    run_parser.add_argument("--output-path", default="")
    run_parser.add_argument("--memory-root", default="")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.subcommand != "run":
        parser.print_help()
        return 0

    objective = (args.objective_flag or args.objective).strip()
    if not objective:
        parser.error("an objective is required")

    context: dict[str, Any] = {
        "working_dir": args.working_dir,
        "constraints": list(args.constraint),
    }
    if args.commands:
        context["commands"] = list(args.commands)
    if args.api_url:
        context["api_requests"] = [{"url": url} for url in args.api_url]
    if args.output_path:
        context["output_path"] = args.output_path

    result = run(
        objective=objective,
        context=context,
        config={
            "max_steps": args.max_steps,
            "mode": args.mode,
            "adapter": args.adapter,
            "memory_root": args.memory_root or None,
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
