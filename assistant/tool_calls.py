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


def _extract_calls_from_list(items: list[Any], calls: list[dict[str, Any]]) -> None:
    """Recursively flatten nested tool_calls arrays into `calls`."""
    for item in items:
        if not isinstance(item, dict):
            continue
        # Nested wrapper: {"tool_calls": [...]} inside the outer list
        if "tool_calls" in item and isinstance(item["tool_calls"], list):
            _extract_calls_from_list(item["tool_calls"], calls)
            continue
        call = _normalize_call(item)
        if call:
            calls.append(call)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    payload = parse_json_payload(text)
    if payload is None:
        return []

    calls: list[dict[str, Any]] = []

    if isinstance(payload, dict):
        if "tool_calls" in payload and isinstance(payload["tool_calls"], list):
            _extract_calls_from_list(payload["tool_calls"], calls)
            return calls

        call = _normalize_call(payload)
        if call:
            calls.append(call)
            return calls

    if isinstance(payload, list):
        _extract_calls_from_list(payload, calls)

    return calls
