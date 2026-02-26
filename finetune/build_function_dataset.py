from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _tool_json_from_calls(calls: list[dict[str, Any]]) -> str:
    payload_calls: list[dict[str, Any]] = []
    for call in calls:
        name = call.get("tool") or call.get("name")
        args = call.get("args", {})
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(args, dict):
            continue
        payload_calls.append({"tool": name.strip(), "args": args})
    return json.dumps({"tool_calls": payload_calls}, ensure_ascii=False)


def _is_function_request(text: str) -> bool:
    t = text.lower()
    markers = (
        "function",
        "write code",
        "implement",
        "custom",
        "utility",
    )
    return any(m in t for m in markers)


def _parse_tool_names(assistant_content: str) -> list[str]:
    try:
        payload = json.loads(assistant_content)
    except Exception:
        return []
    names: list[str] = []
    calls = payload.get("tool_calls") if isinstance(payload, dict) else None
    if not isinstance(calls, list):
        return names
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = call.get("tool")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _is_continuation_turn(user_text: str) -> bool:
    t = user_text.lower()
    return "tool execution is complete" in t or "continue generation" in t


def _passes_tool_quality(user_text: str, assistant_content: str) -> bool:
    names = set(_parse_tool_names(assistant_content))
    if not names:
        if _is_continuation_turn(user_text):
            t = assistant_content.lower()
            return "def " in t or "```python" in t
        return False

    has_find = "find_in_memory" in names
    has_search = "search_web" in names
    has_create = "create_function" in names

    if _is_continuation_turn(user_text):
        return has_create or (has_find and has_search)

    if has_create:
        return has_find

    return has_find and has_search


def normalize_seed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        user = messages[0].get("content", "") if isinstance(messages[0], dict) else ""
        assistant = messages[1].get("content", "") if isinstance(messages[1], dict) else ""
        if not isinstance(user, str) or not isinstance(assistant, str):
            continue
        if not _is_function_request(user):
            continue
        if not _passes_tool_quality(user, assistant):
            continue
        out.append({"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]})
    return out


def normalize_runtime(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages")
        calls = row.get("tool_calls")
        if not isinstance(messages, list) or not isinstance(calls, list):
            continue
        if not messages or not isinstance(messages[0], dict):
            continue
        user = messages[0].get("content", "")
        if not isinstance(user, str) or not _is_function_request(user):
            continue
        assistant = _tool_json_from_calls(calls)
        if assistant == '{"tool_calls": []}':
            continue
        if not _passes_tool_quality(user, assistant):
            continue
        out.append({"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]})
    return out


def normalize_supervision(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        event = row.get("event")
        user = row.get("user_message")
        assistant = row.get("assistant_text")
        if event != "presearch_for_code":
            continue
        if not isinstance(user, str) or not isinstance(assistant, str):
            continue
        if not _is_function_request(user):
            continue
        if not _passes_tool_quality(user, assistant):
            continue
        out.append({"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]})
    return out


def dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for rec in records:
        messages = rec.get("messages", [])
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        user = messages[0].get("content", "") if isinstance(messages[0], dict) else ""
        assistant = messages[1].get("content", "") if isinstance(messages[1], dict) else ""
        if not isinstance(user, str) or not isinstance(assistant, str):
            continue
        key = (user.strip().lower(), assistant.strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build function-focused tool-use fine-tuning dataset.")
    parser.add_argument("--output", default="finetune/train_function_tools.jsonl")
    args = parser.parse_args()

    seed = normalize_seed(iter_jsonl(Path("finetune/tool_use_seed.jsonl")))
    function_seed = normalize_seed(iter_jsonl(Path("finetune/function_creation_tool_use.jsonl")))
    runtime = normalize_runtime(iter_jsonl(Path("memory/tool_finetune_samples.jsonl")))
    supervision = normalize_supervision(iter_jsonl(Path("memory/tool_supervision.jsonl")))

    records = dedupe(seed + function_seed + runtime + supervision)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
