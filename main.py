from __future__ import annotations

import argparse
import os
from pathlib import Path

from assistant.cli_format import print_formatted_output, print_tool_event
from assistant.logging_config import setup_logging
from assistant.chat_engine import ChatEngine
from assistant.functions_registry import FunctionRegistry
from assistant.memory import MemoryStore
from assistant.model import build_model, list_openrouter_models
from assistant.prompt import SYSTEM_PROMPT
from assistant.tools import ToolSystem
from assistant.workspace_tools import WorkspaceTools


def load_dotenv(path: str | Path | None = None) -> None:
    if path is None:
        path = Path(__file__).resolve().parent / ".env"
    env_path = Path(path)
    if not env_path.exists() or not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def build_engine(
    model_name: str,
    provider: str,
    ollama_url: str,
    openrouter_url: str,
    openrouter_api_key: str | None,
    google_api_key: str | None = None,
    autonomous_enabled: bool = False,
    autonomous_steps: int = 6,
) -> ChatEngine:
    root = Path(__file__).resolve().parent

    memory_store = MemoryStore(root / "memory" / "blocks")
    function_registry = FunctionRegistry(root / "functions")
    workspace_tools = WorkspaceTools(root)
    tools = ToolSystem(
        memory_store=memory_store,
        function_registry=function_registry,
        workspace_tools=workspace_tools,
    )
    model = build_model(
        model_name=model_name,
        provider=provider,
        ollama_url=ollama_url,
        openrouter_url=openrouter_url,
        openrouter_api_key=openrouter_api_key,
        google_api_key=google_api_key,
    )

    return ChatEngine(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        autonomous_enabled=autonomous_enabled,
        autonomous_steps=autonomous_steps,
    )


def print_model_connection_info(engine: ChatEngine) -> None:
    model = engine.model
    info_fn = getattr(model, "info", None)
    info = info_fn() if callable(info_fn) else {}
    provider = info.get("provider", "unknown")
    model_name = info.get("model", "")
    endpoint = info.get("endpoint", "")
    native_stream = bool(info.get("native_streaming", False))
    reason = getattr(model, "reason", "")

    print("Model connection:")
    print(f"- provider: {provider}")
    print(f"- model: {model_name}")
    if endpoint:
        print(f"- endpoint: {endpoint}")
    print(f"- streaming: {'native' if native_stream else 'chunked-fallback'}")
    if reason:
        print(f"- status: {reason}")

    connect_log = info.get("connect_log", [])
    details = info.get("details", {})
    if isinstance(details, dict) and details:
        print("- details:")
        for k, v in details.items():
            print(f"  {k}: {v}")
    if isinstance(connect_log, list) and connect_log:
        print("- connect log:")
        for item in connect_log:
            print(f"  {item}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local coding assistant")
    parser.add_argument(
        "--model",
        default=os.getenv("ASSISTANT_MODEL", "arcee-ai/trinity-large-preview:free"),
        help="Model name for selected provider (e.g., qwen3-vl:8b or openai/gpt-4o-mini)",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("ASSISTANT_PROVIDER", "openrouter"),
        choices=["auto", "ollama", "openrouter", "google"],
        help="Model backend provider",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        help="Ollama base URL",
    )
    parser.add_argument(
        "--openrouter-url",
        default=os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1"),
        help="OpenRouter base URL",
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=os.getenv("OPENROUTER_API_KEY", ""),
        help="OpenRouter API key",
    )
    parser.add_argument(
        "--google-api-key",
        default=os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", "")),
        help="Google AI Studio API key (or use GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List OpenRouter models and exit",
    )
    parser.add_argument(
        "--once",
        default="",
        help="Run one user prompt and exit",
    )
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Enable autonomous self-operating mode for each prompt in CLI",
    )
    parser.add_argument(
        "--autonomous-steps",
        type=int,
        default=int(os.getenv("ASSISTANT_AUTONOMOUS_STEPS", "0")),
        help="Max autonomous steps per objective (>0 finite, 0 infinite)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging to stderr",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    setup_logging(verbose=args.verbose)
    if args.list_models:
        models = list_openrouter_models(
            api_key=args.openrouter_api_key,
            base_url=args.openrouter_url,
        )
        if not models:
            print("No models found. Check OPENROUTER_API_KEY / --openrouter-api-key.")
            return
        print("OpenRouter models:")
        for m in models:
            print(f"- {m}")
        return

    engine = build_engine(
        model_name=args.model,
        provider=args.provider,
        ollama_url=args.ollama_url,
        openrouter_url=args.openrouter_url,
        openrouter_api_key=args.openrouter_api_key,
        google_api_key=args.google_api_key,
        autonomous_enabled=args.autonomous,
        autonomous_steps=args.autonomous_steps,
    )
    print_model_connection_info(engine)
    print("")

    if args.once:
        response = engine.handle_turn(args.once, print_tool_event)
        print_formatted_output(response=response)
        return

    engine.run_cli()


if __name__ == "__main__":
    main()
