from __future__ import annotations

import argparse
import os
from pathlib import Path

from assistant.cli_format import print_formatted_output, print_tool_event
from assistant.chat_engine import ChatEngine
from assistant.functions_registry import FunctionRegistry
from assistant.memory import MemoryStore
from assistant.model import build_model
from assistant.prompt import SYSTEM_PROMPT
from assistant.tools import ToolSystem


def build_engine(model_name: str, ollama_url: str) -> ChatEngine:
    root = Path(__file__).resolve().parent

    memory_store = MemoryStore(root / "memory" / "blocks")
    function_registry = FunctionRegistry(root / "functions")
    tools = ToolSystem(memory_store=memory_store, function_registry=function_registry)
    model = build_model(model_name=model_name, base_url=ollama_url)

    return ChatEngine(model=model, tools=tools, system_prompt=SYSTEM_PROMPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local coding assistant")
    parser.add_argument(
        "--model",
        default=os.getenv("ASSISTANT_MODEL", "qwen3-vl:8b"),
        help="Local Ollama model name",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        help="Ollama base URL",
    )
    parser.add_argument(
        "--once",
        default="",
        help="Run one user prompt and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = build_engine(model_name=args.model, ollama_url=args.ollama_url)

    if args.once:
        response = engine.handle_turn(args.once, print_tool_event)
        print_formatted_output(response=response)
        return

    engine.run_cli()


if __name__ == "__main__":
    main()
