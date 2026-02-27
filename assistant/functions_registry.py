from __future__ import annotations

import ast
import hashlib
import io
import json
import tokenize
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .utils import ensure_dir, normalize_keywords, read_json, slugify, utc_now_iso, write_json, write_text


class FunctionRegistry:
    def __init__(self, functions_dir: str | Path = "functions") -> None:
        self.functions_dir = Path(functions_dir)
        ensure_dir(self.functions_dir)

    def _metadata_files(self) -> list[Path]:
        """Return a sorted list of metadata JSON files."""
        return sorted(self.functions_dir.glob("*.json"))

    def _list_metadata(self) -> list[dict[str, Any]]:
        out = []
        for path in self._metadata_files():
            try:
                item = read_json(path)
                item["_meta_path"] = str(path)
                out.append(item)
            except Exception:
                continue
        return out

    def _normalize_code(self, code: str) -> str:
        """Normalize code for hashing by removing comments and whitespace variations."""
        stripped = code.strip()
        if not stripped:
            return ""

        try:
            tree = ast.parse(stripped)
            return ast.unparse(tree).strip()
        except Exception:
            pass

        tokens = []
        reader = io.StringIO(stripped).readline
        try:
            for tok in tokenize.generate_tokens(reader):
                if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                    continue
                tokens.append(tok.string)
        except Exception:
            return "".join(stripped.split())

        return " ".join(tokens)

    def _code_hash(self, code: str) -> str:
        """Generate a SHA-256 hash of the normalized code."""
        normalized = self._normalize_code(code)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _keyword_overlap_ratio(self, a: list[str], b: list[str]) -> float:
        sa = set(normalize_keywords(a))
        sb = set(normalize_keywords(b))
        if not sa or not sb:
            return 0.0
        overlap = len(sa & sb)
        return overlap / min(len(sa), len(sb))

    def _name_similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, slugify(a), slugify(b)).ratio()

    def _find_duplicate(self, name: str, keywords: list[str], code_hash: str) -> dict[str, Any] | None:
        for meta in self._list_metadata():
            existing_hash = meta.get("code_hash", "")
            if existing_hash and existing_hash == code_hash:
                return {
                    "reason": "code_hash_match",
                    "existing_name": meta.get("name"),
                    "meta_path": meta.get("_meta_path"),
                }

            name_sim = self._name_similarity(name, meta.get("name", ""))
            kw_overlap = self._keyword_overlap_ratio(keywords, meta.get("keywords", []))
            if name_sim >= 0.86 and kw_overlap >= 0.6:
                return {
                    "reason": "name_and_keyword_similarity",
                    "existing_name": meta.get("name"),
                    "meta_path": meta.get("_meta_path"),
                    "name_similarity": round(name_sim, 3),
                    "keyword_overlap": round(kw_overlap, 3),
                }
        return None

    def create_function(
        self,
        name: str,
        description: str,
        keywords: list[str],
        code: str = "",
        tool_name: str = "",
        tool_args: dict[str, Any] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        version: str = "1.0.0",
    ) -> dict[str, Any]:
        """
        Register a new function (either raw code or a tool macro).
        
        Args:
            name: Human-readable name.
            description: What the function does.
            keywords: Keywords for discovery.
            code: Python source code (for 'code' kind).
            tool_name: Single tool to call (for 'tool_macro' kind).
            tool_args: Arguments for the single tool.
            tool_calls: Sequence of tool calls (for 'tool_macro' kind).
            version: Semantic version string.
        """
        safe_name = slugify(name)
        py_path = self.functions_dir / f"{safe_name}.py"
        meta_path = self.functions_dir / f"{safe_name}.json"

        kind = "code"
        normalized_tool_calls: list[dict[str, Any]] = []

        if isinstance(tool_calls, list) and tool_calls:
            kind = "tool_macro"
            for item in tool_calls:
                if not isinstance(item, dict):
                    continue
                raw_tool = item.get("tool") or item.get("name")
                tool = str(raw_tool).strip() if isinstance(raw_tool, str) else ""
                if not tool:
                    continue
                args = item.get("args", {})
                if not isinstance(args, dict):
                    args = {}
                normalized_tool_calls.append({"tool": tool, "args": args})

        elif isinstance(tool_name, str) and tool_name.strip():
            kind = "tool_macro"
            normalized_tool_calls = [
                {
                    "tool": tool_name.strip(),
                    "args": tool_args if isinstance(tool_args, dict) else {},
                }
            ]

        if kind == "tool_macro":
            if not normalized_tool_calls:
                return {
                    "ok": False,
                    "created": False,
                    "error": "tool macro requires tool_name or tool_calls",
                }
            rendered_calls = json.dumps(normalized_tool_calls, ensure_ascii=False, indent=2)
            code_text = (
                '"""Registered tool-macro function.\n'
                "Run the listed tool calls in order.\n"
                '"""\n\n'
                f"TOOL_CALLS = {rendered_calls}\n\n"
                "def run(execute_tool):\n"
                '    """execute_tool(name: str, args: dict) -> dict"""\n'
                "    results = []\n"
                "    for call in TOOL_CALLS:\n"
                '        name = call.get("tool", "")\n'
                '        args = call.get("args", {})\n'
                "        results.append(execute_tool(name, args))\n"
                "    return results\n"
            )
        else:
            if not code.strip():
                return {
                    "ok": False,
                    "created": False,
                    "error": "code must not be empty for code function",
                }
            code_text = code.strip() + "\n"

        code_hash = self._code_hash(code_text)

        duplicate = self._find_duplicate(name=name, keywords=keywords, code_hash=code_hash)
        if duplicate:
            return {
                "ok": False,
                "created": False,
                "duplicate": True,
                **duplicate,
            }

        metadata = {
            "name": name,
            "description": description,
            "keywords": normalize_keywords(keywords),
            "version": version,
            "created_at": utc_now_iso(),
            "code_hash": code_hash,
            "kind": kind,
        }
        if normalized_tool_calls:
            metadata["tool_calls"] = normalized_tool_calls

        write_text(py_path, code_text)
        write_json(meta_path, metadata)

        return {
            "ok": True,
            "created": True,
            "function_file": str(py_path),
            "metadata_file": str(meta_path),
            "code_hash": code_hash,
            "kind": kind,
            "tool_calls": normalized_tool_calls,
        }
