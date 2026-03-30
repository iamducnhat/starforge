from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import __version__, create_runtime, run
from .adapters import get_adapter


KNOWN_SUBCOMMANDS = {"run", "adapters", "tools"}

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_CYAN = "\033[36m"


def _style(text: str, *codes: str, enabled: bool) -> str:
    if not enabled or not text:
        return text
    prefix = "".join(codes)
    return f"{prefix}{text}{ANSI_RESET}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="starforge",
        description=(
            "Starforge is a general-purpose research + action agent.\n\n"
            "How to use it:\n"
            "  starforge \"analyze BTC trend and summarize key signals\"\n"
            "  starforge run \"analyze BTC trend and summarize key signals\"\n"
            "  starforge run --working-dir ./repo --command \"pytest -q\" \"debug the failing workflow\"\n"
            "  starforge run --adapter api --api-url https://api.coingecko.com/api/v3/ping \"inspect API connectivity\"\n"
            "  starforge tools\n"
            "  starforge adapters\n\n"
            "Defaults:\n"
            "  - autonomous execution is on by default\n"
            "  - plain objectives are treated as `run`\n"
            "  - `run` is the main execution command\n\n"
            "Common run args:\n"
            "  --working-dir PATH\n"
            "  --adapter {cli,code,api}\n"
            "  --command CMD            repeatable\n"
            "  --api-url URL            repeatable\n"
            "  --constraint TEXT        repeatable\n"
            "  --max-steps N            0 means open-ended autonomous run\n"
            "  --human-readable         print human-readable output\n"
            "  --stream                 stream step progress while running\n"
            "  --output-path PATH\n"
            "  --memory-root PATH\n\n"
            "Use `starforge run --help` for the full execution flag list."
        ),
        epilog=(
            "Examples:\n"
            "  starforge \"find how to use the Binance API and summarize it\"\n"
            "  starforge run \"find how to use the Binance API and summarize it\"\n"
            "  starforge run --working-dir ./repo --command \"pytest -q\" \"debug the failing workflow\"\n"
            "  starforge run --adapter api --api-url https://api.coingecko.com/api/v3/ping \"inspect API connectivity\"\n"
            "  starforge tools --adapter api"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"starforge {__version__}",
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Execute an objective",
        description=(
            "Execute one objective with Starforge.\n\n"
            "Behavior:\n"
            "- autonomous mode is the default\n"
            "- if `--max-steps` is 0, Starforge uses an open-ended step budget\n"
            "- adapters expose different tool environments: `cli`, `code`, `api`"
        ),
        epilog=(
            "Examples:\n"
            "  starforge run \"analyze BTC trend and summarize key signals\"\n"
            "  starforge run --working-dir ./repo --command \"pytest -q\" \"debug the failing tests\"\n"
            "  starforge run --adapter api --api-url https://api.coingecko.com/api/v3/ping \"check API connectivity\""
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    run_parser.add_argument("objective", nargs="?", default="", help="Objective text to execute.")
    run_parser.add_argument(
        "--objective",
        dest="objective_flag",
        default="",
        help="Objective text, as an alternative to the positional argument.",
    )
    run_parser.add_argument(
        "--working-dir",
        default=".",
        help="Working directory for local tools. Default: current directory.",
    )
    run_parser.add_argument(
        "--constraint",
        action="append",
        default=[],
        help="Constraint to apply during execution. Repeat to add multiple constraints.",
    )
    run_parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum tool steps. Default: 0 (open-ended autonomous run).",
    )
    run_parser.add_argument(
        "--mode",
        default="autonomous",
        help="Execution mode. Default: autonomous.",
    )
    run_parser.add_argument(
        "--adapter",
        choices=["cli", "code", "api"],
        default="cli",
        help="Tool environment to expose. Default: cli.",
    )
    run_parser.add_argument(
        "--command",
        dest="commands",
        action="append",
        default=[],
        help="Local command to make available as part of the objective. Repeat to add multiple commands.",
    )
    run_parser.add_argument(
        "--api-url",
        action="append",
        default=[],
        help="API endpoint to seed into the objective context. Repeat to add multiple URLs.",
    )
    run_parser.add_argument(
        "--output-path",
        default="",
        help="Optional output file for generated summaries or artifacts.",
    )
    run_parser.add_argument(
        "--memory-root",
        default="",
        help="Optional override for Starforge memory storage.",
    )
    run_parser.add_argument(
        "--model-feedback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable model-in-the-loop replanning when autonomous queue becomes empty (default: enabled).",
    )
    run_parser.add_argument(
        "--model-name",
        default="",
        help="Optional model name override for autonomous model feedback.",
    )
    run_parser.add_argument(
        "--model-provider",
        default="",
        help="Optional provider override for autonomous model feedback (auto/ollama/openrouter/google/nvidia).",
    )
    run_parser.add_argument(
        "--model-orchestrated",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require tool actions to be model-generated when model feedback is available.",
    )
    run_parser.add_argument(
        "--human-readable",
        "--human",
        action="store_true",
        help="Print human-readable output instead of JSON.",
    )
    run_parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream step progress while running (default: enabled).",
    )
    run_parser.add_argument(
        "--ansi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ANSI colors for human-readable output (default: enabled).",
    )

    adapters_parser = subparsers.add_parser(
        "adapters",
        help="List adapters",
        description="List available adapters and what each one is for.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    adapters_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )

    tools_parser = subparsers.add_parser(
        "tools",
        help="List tools",
        description="List built-in tools, optionally filtered by adapter.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    tools_parser.add_argument(
        "--adapter",
        choices=["cli", "code", "api"],
        default="cli",
        help="Adapter whose toolset should be shown. Default: cli.",
    )
    tools_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )

    return parser


def _adapter_payload() -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for name in ("cli", "code", "api"):
        adapter = get_adapter(name)
        payload.append(
            {
                "name": adapter.name,
                "description": adapter.description,
                "tools": [tool.name for tool in adapter.tools],
            }
        )
    return payload


def _print_adapters(json_mode: bool) -> int:
    payload = _adapter_payload()
    if json_mode:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print("Available adapters:\n")
    for item in payload:
        print(f"{item['name']}: {item['description']}")
        print(f"  tools: {', '.join(item['tools'])}")
    return 0


def _print_tools(adapter_name: str, json_mode: bool) -> int:
    runtime = create_runtime(adapter=adapter_name)
    payload = {
        "adapter": adapter_name,
        "tools": runtime.registry.describe(),
    }
    if json_mode:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"Tools for adapter '{adapter_name}':\n")
    for item in payload["tools"]:
        print(f"- {item['name']}: {item['description']}")
    return 0


def _print_run_result(result: dict[str, Any], human_mode: bool, *, ansi_enabled: bool = False) -> None:
    if not human_mode:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    def _truncate(value: Any, limit: int = 180) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    def _compact_json(value: Any, limit: int = 220) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            text = str(value)
        return _truncate(text, limit=limit)

    def _observation_summary(observation: dict[str, Any]) -> str:
        obs_type = str(observation.get("type", "unknown"))
        content = observation.get("content")
        metadata = observation.get("metadata") if isinstance(observation.get("metadata"), dict) else {}

        if obs_type == "filesystem_snapshot":
            count = metadata.get("count")
            root = metadata.get("root")
            return _truncate(f"count={count}, root={root}" if count is not None else f"root={root}")
        if obs_type == "file_read":
            path = metadata.get("path")
            bytes_count = metadata.get("bytes")
            return _truncate(
                f"path={path}, bytes={bytes_count}" if bytes_count is not None else f"path={path}",
            )
        if obs_type == "file_write":
            path = metadata.get("path")
            bytes_count = None
            if isinstance(content, dict):
                bytes_count = content.get("bytes")
            return _truncate(
                f"path={path}, bytes={bytes_count}" if bytes_count is not None else f"path={path}",
            )
        if obs_type == "command_result":
            if isinstance(content, dict):
                command = content.get("command")
                exit_code = content.get("exit_code")
                signal = str(content.get("stderr") or content.get("stdout") or "").strip()
                signal_text = _truncate(signal.replace("\n", " "), limit=120)
                return _truncate(f"exit={exit_code}, command={command}, signal={signal_text}", limit=260)
            return _truncate(content, limit=260)
        if obs_type == "search_results":
            query = metadata.get("query")
            count = metadata.get("count")
            top_title = ""
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict):
                    top_title = str(first.get("title", "")).strip()
            parts = [f"query={query}", f"count={count}"]
            if top_title:
                parts.append(f"top={top_title}")
            return _truncate(", ".join(parts), limit=260)
        if obs_type == "webpage_read":
            title = metadata.get("title")
            url = metadata.get("url")
            return _truncate(f"title={title}, url={url}", limit=260)
        if obs_type == "api_response":
            status_code = metadata.get("status_code")
            url = metadata.get("url")
            return _truncate(f"status={status_code}, url={url}", limit=260)
        if obs_type == "tool_error":
            return _truncate(content, limit=260)
        return _truncate(content, limit=260)

    payload = dict(result.get("result") or {})
    human_summary = str(payload.get("human_readable") or "").strip()
    if human_summary:
        print(human_summary)
        print("")
    success = bool(result.get("success", False))
    success_text = _style(str(success), ANSI_GREEN if success else ANSI_RED, ANSI_BOLD, enabled=ansi_enabled)
    print(f"{_style('Success', ANSI_BOLD, enabled=ansi_enabled)}: {success_text}")
    print(f"{_style('Steps', ANSI_BOLD, enabled=ansi_enabled)}: {int(result.get('steps', 0) or 0)}")
    print(f"{_style('Confidence', ANSI_BOLD, enabled=ansi_enabled)}: {result.get('confidence', 0.0)}")
    model_audit = payload.get("model_audit") if isinstance(payload.get("model_audit"), dict) else {}
    if model_audit:
        print("")
        print(_style("Completion audit:", ANSI_BOLD, ANSI_YELLOW, enabled=ansi_enabled))
        print(f"- pass: {bool(model_audit.get('pass'))}")
        if model_audit.get("score") is not None:
            print(f"- score: {model_audit.get('score')}")
        if model_audit.get("threshold") is not None:
            print(f"- threshold: {model_audit.get('threshold')}")
        result_text = str(model_audit.get("result", "")).strip()
        if result_text:
            print(f"- result: {_truncate(result_text, limit=200)}")
        if model_audit.get("cannot_improve") is not None:
            print(f"- cannot_improve: {bool(model_audit.get('cannot_improve'))}")

    actions = result.get("actions") if isinstance(result.get("actions"), list) else []
    if actions:
        print("")
        print(_style("Execution plan (from step rationale):", ANSI_BOLD, ANSI_YELLOW, enabled=ansi_enabled))
        for index, action in enumerate(actions, start=1):
            item = dict(action) if isinstance(action, dict) else {}
            tool_name = str(item.get("tool", "unknown"))
            rationale = str(item.get("rationale", "")).strip()
            tool_label = _style(tool_name, ANSI_CYAN, ANSI_BOLD, enabled=ansi_enabled)
            if rationale:
                print(f"{index}. {tool_label}: {rationale}")
            else:
                print(f"{index}. {tool_label}")

    if not actions:
        return
    print("")
    print(_style("Actions:", ANSI_BOLD, ANSI_YELLOW, enabled=ansi_enabled))
    for index, action in enumerate(actions, start=1):
        item = dict(action) if isinstance(action, dict) else {}
        observation = item.get("observation") if isinstance(item.get("observation"), dict) else {}
        observation_type = str(observation.get("type", "unknown"))
        tool_name = str(item.get("tool", "unknown"))
        status = str(item.get("status", "unknown"))
        status_color = ANSI_GREEN if status.casefold() == "completed" else ANSI_RED if status.casefold() == "failed" else ""
        tool_label = _style(tool_name, ANSI_CYAN, ANSI_BOLD, enabled=ansi_enabled)
        status_label = _style(status, status_color, ANSI_BOLD, enabled=ansi_enabled)
        observation_label = _style(observation_type, ANSI_CYAN, enabled=ansi_enabled)
        print(f"{index}. {tool_label} [{status_label}] -> {observation_label}")
        arguments = item.get("arguments")
        if isinstance(arguments, dict) and arguments:
            print(f"   args: {_compact_json(arguments)}")
        rationale = str(item.get("rationale", "")).strip()
        if rationale:
            print(f"   why: {rationale}")
        summary = _observation_summary(observation)
        if summary:
            print(f"   result: {summary}")

    terminal_logs: list[tuple[str, str]] = []
    for action in actions:
        item = dict(action) if isinstance(action, dict) else {}
        observation = item.get("observation") if isinstance(item.get("observation"), dict) else {}
        if str(observation.get("type")) != "command_result":
            continue
        content = observation.get("content")
        if not isinstance(content, dict):
            continue
        command = str(content.get("command") or "").strip() or "<unknown command>"
        merged_output = str(content.get("stdout") or "").strip()
        stderr = str(content.get("stderr") or "").strip()
        if stderr:
            merged_output = f"{merged_output}\n{stderr}".strip() if merged_output else stderr
        if merged_output:
            terminal_logs.append((command, merged_output))
    if terminal_logs:
        print("")
        print(_style("Terminal output log:", ANSI_BOLD, ANSI_YELLOW, enabled=ansi_enabled))
        for index, (command, output) in enumerate(terminal_logs, start=1):
            print(f"{index}. command: {command}")
            print("   output:")
            for line in output.splitlines()[:20]:
                print(f"     {line}")


def _build_stream_callback(*, human_mode: bool, ansi_enabled: bool = False):
    def _truncate(value: Any, limit: int = 120) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    def _compact_json(value: Any, limit: int = 140) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            text = str(value)
        return _truncate(text, limit=limit)

    def _on_action(event: dict[str, Any]) -> None:
        index = event.get("index", "?")
        tool_name = str(event.get("tool", "unknown"))
        status = str(event.get("status", "unknown"))
        observation_type = str(event.get("observation_type", "unknown"))
        status_color = ANSI_GREEN if status.casefold() == "completed" else ANSI_RED if status.casefold() == "failed" else ""
        step_label = _style(f"[step {index}]", ANSI_CYAN, ANSI_BOLD, enabled=ansi_enabled)
        tool_label = _style(tool_name, ANSI_CYAN, ANSI_BOLD, enabled=ansi_enabled)
        status_label = _style(status, status_color, ANSI_BOLD, enabled=ansi_enabled)
        observation_label = _style(observation_type, ANSI_CYAN, enabled=ansi_enabled)
        line = f"{step_label} {tool_label} [{status_label}] -> {observation_label}"
        arguments = event.get("arguments")
        if isinstance(arguments, dict) and arguments:
            line += f" | args={_compact_json(arguments)}"
        rationale = str(event.get("rationale", "")).strip()
        if rationale:
            line += f" | why={_truncate(rationale)}"
        if human_mode:
            print(line, flush=True)
        else:
            print(line, file=sys.stderr, flush=True)

    return _on_action


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    if argv and argv[0] not in KNOWN_SUBCOMMANDS and not str(argv[0]).startswith("-"):
        argv = ["run", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "adapters":
        return _print_adapters(json_mode=bool(args.json))

    if args.subcommand == "tools":
        return _print_tools(adapter_name=str(args.adapter), json_mode=bool(args.json))

    if args.subcommand != "run":
        parser.print_help()
        return 0

    human_mode = bool(args.human_readable)
    ansi_enabled = bool(args.ansi and human_mode)

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
            "model_feedback": bool(args.model_feedback),
            "model_name": args.model_name or None,
            "feedback_provider": args.model_provider or None,
            "model_orchestrated": bool(args.model_orchestrated),
            "on_action": _build_stream_callback(human_mode=human_mode, ansi_enabled=ansi_enabled)
            if bool(args.stream)
            else None,
        },
    )
    _print_run_result(result=result, human_mode=human_mode, ansi_enabled=ansi_enabled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
