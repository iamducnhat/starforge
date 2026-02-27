from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _tool_payload(calls: list[dict[str, Any]]) -> str:
    return json.dumps({"tool_calls": calls}, ensure_ascii=False)


def _keywords_from_topic(topic: str, limit: int = 6) -> list[str]:
    words = [w.strip(" ,.-").lower() for w in topic.split()]
    words = [w for w in words if w and len(w) >= 3]
    dedup: list[str] = []
    seen = set()
    for w in words:
        if w not in seen:
            dedup.append(w)
            seen.add(w)
        if len(dedup) >= limit:
            break
    return dedup or ["general"]


def _mk_record(ts: datetime, user: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "created_at": ts.isoformat(),
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": _tool_payload(calls)},
        ],
    }


def build_records(per_topic: int = 120, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    now = datetime(2026, 2, 27, tzinfo=timezone.utc)
    records: list[dict[str, Any]] = []

    factual_topics = [
        "Python coding assistant best practices architecture design patterns",
        "OpenRouter model latency and streaming troubleshooting",
        "Ollama context window and token generation settings",
        "Vietnam travel overview and food recommendations",
        "LoRA fine-tuning best practices for small models",
        "CLI markdown ANSI rendering in terminal apps",
        "AI coding agent tool-calling reliability",
        "local RAG memory design with filesystem blocks",
    ]
    factual_time_prefixes = [
        "latest",
        "current",
        "today",
        "newest",
        "recent",
    ]
    factual_frames = [
        "Can you explain {prefix} {topic} in 2026?",
        "I need {prefix} info about {topic}.",
        "Find {prefix} updates for {topic}.",
        "What are {prefix} trends for {topic}?",
    ]

    coding_topics = [
        "download file in python safely",
        "extract text from pdf python",
        "stream tokens from model api python",
        "edit files with path guard in python",
        "auto compact context window logic python",
        "tool call parser robust json python",
    ]
    coding_frames = [
        "Create a reusable function for {topic}.",
        "Write Python code to {topic}.",
        "Build helper utility for {topic}.",
        "Implement clean function for {topic}.",
    ]

    workspace_frames = [
        "In this repo, update CLI commands and fix parser bugs.",
        "Please modify project files to add a new tool and wire it.",
        "Refactor chat engine in this codebase and adjust prompt.",
        "Add a command to main CLI and update README.",
    ]
    create_file_frames = [
        "Create a new file assistant/{name}.py and add starter code.",
        "Please add new module assistant/{name}.py for this feature.",
    ]
    module_names = [
        "runtime_flags",
        "tool_router",
        "context_compactor",
        "plan_manager",
        "search_index",
    ]

    store_frames = [
        "Save this workflow as reusable function for project research.",
        "Register a tool macro function for web research then summarize.",
        "Store a reusable function to search docs and read pages.",
    ]

    ts = now - timedelta(minutes=50000)

    # Factual (time-sensitive): always include get_current_datetime
    for topic in factual_topics:
        for _ in range(per_topic):
            prefix = rng.choice(factual_time_prefixes)
            frame = rng.choice(factual_frames)
            query = f"{prefix} {topic} 2026"
            user = frame.format(prefix=prefix, topic=topic)
            calls = [
                {"tool": "get_current_datetime", "args": {}},
                {"tool": "find_in_memory", "args": {"keywords": _keywords_from_topic(topic)}},
                {"tool": "search_web", "args": {"query": query, "level": "auto"}},
            ]
            records.append(_mk_record(ts, user, calls))
            ts += timedelta(seconds=1)

    # Code generation presearch
    for topic in coding_topics:
        for _ in range(per_topic):
            frame = rng.choice(coding_frames)
            user = frame.format(topic=topic)
            calls = [
                {"tool": "find_in_memory", "args": {"keywords": _keywords_from_topic(topic)}},
                {
                    "tool": "search_web",
                    "args": {"query": f"how to {topic} best practice", "level": "deep"},
                },
            ]
            records.append(_mk_record(ts, user, calls))
            ts += timedelta(seconds=1)

    # Workspace edit tasks: inspect project first
    for _ in range(per_topic * 3):
        user = rng.choice(workspace_frames)
        calls = [
            {"tool": "list_files", "args": {"path": ".", "max_entries": 200}},
            {"tool": "search_project", "args": {"query": "chat engine tool", "path": ".", "max_matches": 80}},
            {"tool": "read_file", "args": {"path": "assistant/chat_engine.py", "max_chars": 8000}},
            {
                "tool": "edit_file",
                "args": {
                    "path": "assistant/chat_engine.py",
                    "find_text": "Commands:",
                    "replace_text": "Commands:",
                    "replace_all": False,
                },
            },
        ]
        records.append(_mk_record(ts, user, calls))
        ts += timedelta(seconds=1)

    # Create-file tasks
    for _ in range(per_topic * 2):
        name = rng.choice(module_names)
        user = rng.choice(create_file_frames).format(name=name)
        file_path = f"assistant/{name}.py"
        calls = [
            {"tool": "list_files", "args": {"path": "assistant", "max_entries": 200}},
            {
                "tool": "create_file",
                "args": {"path": file_path, "content": '"""Module."""\n\n', "overwrite": False},
            },
            {
                "tool": "edit_file",
                "args": {
                    "path": file_path,
                    "find_text": '"""Module."""',
                    "replace_text": '"""Runtime module."""',
                    "replace_all": False,
                },
            },
        ]
        records.append(_mk_record(ts, user, calls))
        ts += timedelta(seconds=1)

    # Save as tool-macro function
    for _ in range(per_topic * 2):
        user = rng.choice(store_frames)
        tool_workflow = [
            {"tool": "search_web", "args": {"query": "python architecture patterns", "level": "deep"}},
            {"tool": "read_web", "args": {"url": "https://docs.python.org/3/", "max_chars": 8000}},
        ]
        calls = [
            {"tool": "find_in_memory", "args": {"keywords": ["workflow", "reusable", "tool", "search"]}},
            {"tool": "search_web", "args": {"query": "how to design reusable research workflow", "level": "auto"}},
            {
                "tool": "create_function",
                "args": {
                    "name": "research_workflow_macro",
                    "description": "Reusable tool-call workflow for online research",
                    "keywords": ["workflow", "research", "tool", "web"],
                    "tool_calls": tool_workflow,
                },
            },
        ]
        records.append(_mk_record(ts, user, calls))
        ts += timedelta(seconds=1)

    rng.shuffle(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate large synthetic tool-use fine-tuning dataset.")
    parser.add_argument("--output", default="finetune/synthetic_tool_use.jsonl")
    parser.add_argument("--per-topic", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = build_records(per_topic=args.per_topic, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} synthetic records to {out}")


if __name__ == "__main__":
    main()
