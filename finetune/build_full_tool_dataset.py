from __future__ import annotations

import argparse
import json
import random
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


def _to_pair(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    user = messages[0].get("content", "") if isinstance(messages[0], dict) else ""
    assistant = messages[1].get("content", "") if isinstance(messages[1], dict) else ""
    if not isinstance(user, str) or not isinstance(assistant, str):
        return None
    if not user.strip() or not assistant.strip():
        return None
    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


def _tool_json_from_calls(calls: list[dict[str, Any]]) -> str:
    out_calls: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        name = call.get("tool") or call.get("name")
        args = call.get("args", {})
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(args, dict):
            args = {}
        out_calls.append({"tool": name.strip(), "args": args})
    return json.dumps({"tool_calls": out_calls}, ensure_ascii=False)


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
        if not isinstance(user, str) or not user.strip():
            continue
        assistant = _tool_json_from_calls(calls)
        if assistant == '{"tool_calls": []}':
            continue
        out.append(
            {
                "messages": [
                    {"role": "user", "content": user.strip()},
                    {"role": "assistant", "content": assistant},
                ]
            }
        )
    return out


def dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for rec in records:
        pair = _to_pair(rec)
        if not pair:
            continue
        user = pair["messages"][0]["content"]
        assistant = pair["messages"][1]["content"]
        key = (user.lower(), assistant)
        if key in seen:
            continue
        seen.add(key)
        out.append(pair)
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full tool-use fine-tuning dataset.")
    parser.add_argument("--train-output", default="finetune/train_tool_use_full.jsonl")
    parser.add_argument("--val-output", default="finetune/val_tool_use_full.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sources = [
        iter_jsonl(Path("finetune/tool_use_seed.jsonl")),
        iter_jsonl(Path("finetune/function_creation_tool_use.jsonl")),
        iter_jsonl(Path("finetune/synthetic_tool_use.jsonl")),
        iter_jsonl(Path("finetune/train_function_tools.jsonl")),
    ]

    runtime = normalize_runtime(iter_jsonl(Path("memory/tool_finetune_samples.jsonl")))

    merged: list[dict[str, Any]] = []
    for src in sources:
        merged.extend(src)
    merged.extend(runtime)

    rows = dedupe(merged)
    rnd = random.Random(args.seed)
    rnd.shuffle(rows)

    val_count = int(len(rows) * max(0.0, min(args.val_ratio, 0.5)))
    val = rows[:val_count]
    train = rows[val_count:]

    train_path = Path(args.train_output)
    val_path = Path(args.val_output)
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    print(f"Total: {len(rows)} | Train: {len(train)} | Val: {len(val)}")
    print(f"- train: {train_path}")
    print(f"- val:   {val_path}")


if __name__ == "__main__":
    main()
