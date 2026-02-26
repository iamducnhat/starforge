from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\- ]+", "", name).strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    return s or "item"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def _extract_json_code_block(text: str) -> list[str]:
    blocks = []
    pattern = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    for m in pattern.finditer(text):
        candidate = m.group(1).strip()
        if candidate:
            blocks.append(candidate)
    return blocks


def _extract_balanced_json(text: str) -> list[str]:
    candidates = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : i + 1])
                        break
            start = text.find(opener, start + 1)
    return candidates


def parse_json_payload(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped:
        return None

    # Try direct parse first.
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for candidate in _extract_json_code_block(stripped):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    for candidate in _extract_balanced_json(stripped):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def normalize_keywords(keywords: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for kw in keywords:
        cleaned = kw.strip().lower()
        if cleaned and cleaned not in seen:
            out.append(cleaned)
            seen.add(cleaned)
    return out


def short_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)
