from __future__ import annotations

from typing import Any

from .utils import parse_json_payload


def _normalize_call(item: dict[str, Any]) -> dict[str, Any] | None:
    name = item.get("tool") or item.get("name")
    if not isinstance(name, str) or not name.strip():
        return None

    args = item.get("args")
    if args is None:
        args = item.get("arguments", {})

    if not isinstance(args, dict):
        return None

    return {"name": name.strip(), "args": args}


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    payload = parse_json_payload(text)
    if payload is None:
        return []

    calls: list[dict[str, Any]] = []

    if isinstance(payload, dict):
        if "tool_calls" in payload and isinstance(payload["tool_calls"], list):
            for item in payload["tool_calls"]:
                if isinstance(item, dict):
                    call = _normalize_call(item)
                    if call:
                        calls.append(call)
            return calls

        call = _normalize_call(payload)
        if call:
            calls.append(call)
            return calls

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                call = _normalize_call(item)
                if call:
                    calls.append(call)

    return calls
