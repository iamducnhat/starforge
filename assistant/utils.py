from __future__ import annotations

import json
import os
import re
import tempfile
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
    s = s.strip("_")
    return s or "item"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    """Write text to a file atomically using a temporary file."""
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Create a temporary file in the same directory to ensure atomic rename
    fd, tmp_path = tempfile.mkstemp(dir=parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        # os.replace is atomic on both POSIX and Windows (Python 3.3+)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        raise


def write_json(path: Path, data: Any) -> None:
    """Write data to a JSON file atomically."""
    content = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    write_text(path, content)


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


def _heal_json_string(s: str) -> str:
    s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

    out = []
    i = 0
    n = len(s)
    in_string = False
    escape = False

    while i < n:
        c = s[i]

        if not in_string:
            if c == '"':
                in_string = True
            out.append(c)
        else:
            if escape:
                out.append("\\")
                out.append(c)
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                j = i + 1
                while j < n and s[j] in " \n\r\t":
                    j += 1

                is_end = False
                if j == n:
                    is_end = True
                elif s[j] in ",}]":
                    is_end = True
                elif s[j] == ":":
                    k = j + 1
                    while k < n and s[k] in " \n\r\t":
                        k += 1
                    if k < n and s[k] in '"{[tfn0123456789-+':
                        is_end = True

                if is_end:
                    in_string = False
                    out.append(c)
                else:
                    out.append('\\"')
            else:
                out.append(c)
        i += 1

    healed = "".join(out)

    stack = []
    in_str = False
    esc = False
    final_out = []

    for c in healed:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            final_out.append(c)
        else:
            if c == '"':
                in_str = True
                final_out.append(c)
            elif c in "{[":
                stack.append(c)
                final_out.append(c)
            elif c == "}":
                while stack and stack[-1] != "{":
                    stack.pop()
                    final_out.append("]")
                if stack and stack[-1] == "{":
                    stack.pop()
                    final_out.append("}")
            elif c == "]":
                while stack and stack[-1] != "[":
                    stack.pop()
                    final_out.append("}")
                if stack and stack[-1] == "[":
                    stack.pop()
                    final_out.append("]")
            else:
                final_out.append(c)

    if in_str:
        final_out.append('"')

    while stack:
        last = stack.pop()
        final_out.append("}" if last == "{" else "]")

    return "".join(final_out)


def parse_json_payload(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped:
        return None

    # Try direct parse first.
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    try:
        healed = _heal_json_string(stripped)
        return json.loads(healed)
    except json.JSONDecodeError:
        pass

    for candidate in _extract_json_code_block(stripped):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return json.loads(_heal_json_string(candidate))
            except json.JSONDecodeError:
                continue

    for candidate in _extract_balanced_json(stripped):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return json.loads(_heal_json_string(candidate))
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


def get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        return default


def get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "off", "no"}


_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|token|secret|password|passwd|authorization|bearer|access[_-]?key)",
    re.IGNORECASE,
)


def redact_secrets_text(text: str) -> str:
    if not text:
        return text

    out = text
    # KEY=value style
    out = re.sub(
        r"(?im)^(\s*[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|PASSWD|ACCESS_KEY)[A-Z0-9_]*\s*=\s*)(.+)\s*$",
        r"\1***REDACTED***",
        out,
    )
    # Inline key=value style
    out = re.sub(
        r"(?i)\b(api[_-]?key|token|secret|password|passwd|access[_-]?key)\s*=\s*([^\s,;]+)",
        lambda m: f"{m.group(1)}=***REDACTED***",
        out,
    )
    # JSON-like "key": "value"
    out = re.sub(
        r'(?i)"([^"]*(?:api[_-]?key|token|secret|password|passwd|authorization|bearer|access[_-]?key)[^"]*)"\s*:\s*"([^"]*)"',
        r'"\1":"***REDACTED***"',
        out,
    )
    # Bearer tokens in free text
    out = re.sub(r"(?i)\b(Bearer)\s+[A-Za-z0-9._\-=/+]+", r"\1 ***REDACTED***", out)
    # sk- style keys
    out = re.sub(r"\b(sk-[A-Za-z0-9][A-Za-z0-9_\-]{10,})\b", "***REDACTED***", out)
    return out


def redact_secrets_obj(data: Any) -> Any:
    if isinstance(data, dict):
        redacted = {}
        for k, v in data.items():
            key = str(k)
            if _SECRET_KEY_RE.search(key):
                redacted[k] = "***REDACTED***"
            else:
                redacted[k] = redact_secrets_obj(v)
        return redacted
    if isinstance(data, list):
        return [redact_secrets_obj(x) for x in data]
    if isinstance(data, str):
        return redact_secrets_text(data)
    return data
