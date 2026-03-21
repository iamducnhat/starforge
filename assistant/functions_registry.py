from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import tokenize
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from .utils import (ensure_dir, normalize_keywords, read_json, slugify,
                    utc_now_iso, write_json, write_text)


class FunctionRegistry:
    def __init__(self, functions_dir: str | Path = "functions") -> None:
        self.functions_dir = Path(functions_dir)
        ensure_dir(self.functions_dir)
        self.skills_index_path = self.functions_dir / "_skills.json"
        self._skills_cache: dict[str, dict[str, Any]] = {}
        self._load_skills_index()

    def _load_skills_index(self) -> None:
        try:
            data = read_json(self.skills_index_path)
            if isinstance(data, dict):
                self._skills_cache = {
                    str(k): v for k, v in data.items() if isinstance(v, dict)
                }
            else:
                self._skills_cache = {}
        except Exception:
            self._skills_cache = {}

    def _save_skills_index(self) -> None:
        try:
            write_json(self.skills_index_path, self._skills_cache)
        except Exception:
            pass

    @staticmethod
    def _parse_iso_to_utc(value: str) -> datetime | None:
        raw = (value or "").strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except Exception:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _skill_rank(self, skill: dict[str, Any]) -> float:
        success = float(skill.get("success_count", 0.0) or 0.0)
        failure = float(skill.get("failure_count", 0.0) or 0.0)
        confidence_sum = float(skill.get("confidence_sum", 0.0) or 0.0)
        total = max(1.0, success + failure)
        ratio = success / total
        confidence = confidence_sum / total
        last_success = self._parse_iso_to_utc(str(skill.get("last_success_at", "")))
        recency_bonus = 0.0
        if last_success is not None:
            age_days = max(
                0.0, (datetime.now(timezone.utc) - last_success).total_seconds() / 86400.0
            )
            recency_bonus = 0.35 / (1.0 + (age_days / 30.0))
        return round((ratio * 0.9) + (confidence * 0.4) + recency_bonus, 4)

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

    def _paths_for_name(self, name: str) -> tuple[Path, Path]:
        """Return (py_path, meta_path) for a given logical function name."""
        safe_name = slugify(name)
        py_path = self.functions_dir / f"{safe_name}.py"
        meta_path = self.functions_dir / f"{safe_name}.json"
        return py_path, meta_path

    def get_function_metadata(self, name: str) -> dict[str, Any] | None:
        """
        Look up function metadata by logical name.

        First tries slug-based filename, then falls back to scanning all
        metadata entries for a matching "name" field.
        """
        _py, meta_path = self._paths_for_name(name)
        if meta_path.exists():
            try:
                meta = read_json(meta_path)
                meta["_meta_path"] = str(meta_path)
                return meta
            except Exception:
                pass

        for item in self._list_metadata():
            if str(item.get("name", "")).strip() == str(name).strip():
                return item
        return None

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
                if tok.type in (
                    tokenize.COMMENT,
                    tokenize.NL,
                    tokenize.NEWLINE,
                    tokenize.INDENT,
                    tokenize.DEDENT,
                ):
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

    def _find_duplicate(
        self, name: str, keywords: list[str], code_hash: str
    ) -> dict[str, Any] | None:
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
            rendered_calls = json.dumps(
                normalized_tool_calls, ensure_ascii=False, indent=2
            )
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

        duplicate = self._find_duplicate(
            name=name, keywords=keywords, code_hash=code_hash
        )
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

    def create_skill(
        self,
        name: str,
        description: str,
        keywords: list[str],
        code: str = "",
        tool_name: str = "",
        tool_args: dict[str, Any] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        result = self.create_function(
            name=name,
            description=description,
            keywords=keywords,
            code=code,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_calls=tool_calls,
        )
        function_name = name
        if result.get("duplicate"):
            existing = str(result.get("existing_name", "")).strip()
            if existing:
                function_name = existing
        elif not result.get("ok", False):
            return result

        skill_id = slugify(name)
        now = utc_now_iso()
        existing = self._skills_cache.get(skill_id, {})
        entry = {
            "id": skill_id,
            "name": name,
            "description": description,
            "keywords": normalize_keywords(keywords),
            "function_name": function_name,
            "created_at": existing.get("created_at", now),
            "updated_at": now,
            "success_count": int(existing.get("success_count", 0) or 0),
            "failure_count": int(existing.get("failure_count", 0) or 0),
            "confidence_sum": float(existing.get("confidence_sum", 0.0) or 0.0),
            "last_used_at": existing.get("last_used_at", ""),
            "last_success_at": existing.get("last_success_at", ""),
        }
        self._skills_cache[skill_id] = entry
        self._save_skills_index()
        return {
            "ok": True,
            "created": True,
            "skill": entry,
            "function_result": result,
        }

    def list_skills(
        self, limit: int = 50, query: str = "", min_score: float = 0.0
    ) -> dict[str, Any]:
        q = (query or "").strip().lower()
        rows: list[dict[str, Any]] = []
        for skill in self._skills_cache.values():
            name = str(skill.get("name", ""))
            description = str(skill.get("description", ""))
            keywords = skill.get("keywords", [])
            searchable = " ".join(
                [name.lower(), description.lower()]
                + [str(k).lower() for k in keywords if isinstance(k, str)]
            )
            if q and q not in searchable:
                continue
            score = self._skill_rank(skill)
            if score < float(min_score):
                continue
            row = dict(skill)
            row["score"] = score
            rows.append(row)
        rows.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return {"ok": True, "count": len(rows[: max(1, limit)]), "skills": rows[: max(1, limit)]}

    def find_skills(self, query: str, limit: int = 5) -> dict[str, Any]:
        return self.list_skills(limit=limit, query=query, min_score=0.0)

    def record_skill_outcome(
        self,
        name: str,
        success: bool,
        confidence: float = 1.0,
        notes: str = "",
    ) -> dict[str, Any]:
        skill_id = slugify(name)
        skill = self._skills_cache.get(skill_id)
        if skill is None:
            # Try exact name fallback
            for item in self._skills_cache.values():
                if str(item.get("name", "")).strip() == str(name).strip():
                    skill = item
                    skill_id = str(item.get("id", skill_id))
                    break
        if skill is None:
            return {"ok": False, "error": f"skill not found: {name}"}

        if success:
            skill["success_count"] = int(skill.get("success_count", 0) or 0) + 1
            skill["last_success_at"] = utc_now_iso()
        else:
            skill["failure_count"] = int(skill.get("failure_count", 0) or 0) + 1
        conf = max(0.0, min(float(confidence), 1.0))
        skill["confidence_sum"] = float(skill.get("confidence_sum", 0.0) or 0.0) + conf
        skill["last_used_at"] = utc_now_iso()
        skill["updated_at"] = utc_now_iso()
        if notes.strip():
            skill["last_notes"] = notes.strip()[:600]
        self._skills_cache[skill_id] = skill
        self._save_skills_index()
        return {"ok": True, "skill": skill, "score": self._skill_rank(skill)}

    def _record_skill_usage_for_function(self, function_name: str) -> None:
        changed = False
        now = utc_now_iso()
        for skill in self._skills_cache.values():
            if str(skill.get("function_name", "")).strip() != str(function_name).strip():
                continue
            skill["last_used_at"] = now
            skill["updated_at"] = now
            changed = True
        if changed:
            self._save_skills_index()

    def _execute_code_function(self, name: str, args: dict[str, Any]) -> Any:
        """
        Execute a 'code' kind function by importing its module and calling
        a callable that matches the slugified name.
        """
        py_path, _meta_path = self._paths_for_name(name)
        if not py_path.exists():
            raise FileNotFoundError(f"function implementation not found: {name}")

        module_name = f"_assistant_function_{slugify(name)}"
        spec = importlib.util.spec_from_file_location(module_name, py_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load function module for: {name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        func_name = slugify(name)
        fn = getattr(module, func_name, None)
        if not callable(fn):
            raise AttributeError(
                f"callable '{func_name}' not found in function module for: {name}"
            )
        return fn(**(args or {}))

    def execute_function(
        self,
        name: str,
        args: dict[str, Any] | None,
        execute_tool: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Execute a previously registered function by logical name.

        For 'tool_macro' functions, this replays the saved tool_calls
        in order, using the provided execute_tool callback.

        For 'code' functions, this imports the Python module and calls
        a function whose name matches the slugified function name,
        passing args as keyword arguments.
        """
        metadata = self.get_function_metadata(name)
        if not metadata:
            return {"ok": False, "error": f"function not found: {name}"}
        self._record_skill_usage_for_function(name)

        kind = metadata.get("kind", "code")
        args = args or {}

        if kind == "tool_macro":
            calls = metadata.get("tool_calls", []) or []
            if not isinstance(calls, list):
                return {
                    "ok": False,
                    "error": f"invalid tool_calls metadata for function: {name}",
                }
            results: list[dict[str, Any]] = []
            for call in calls:
                if not isinstance(call, dict):
                    continue
                tool = str(call.get("tool", "")).strip()
                call_args = call.get("args", {})
                if not isinstance(call_args, dict):
                    call_args = {}
                if not tool:
                    continue
                results.append(execute_tool(tool, call_args))
            return {
                "ok": True,
                "kind": "tool_macro",
                "results": results,
            }

        try:
            result = self._execute_code_function(name, args)
        except Exception as e:
            return {"ok": False, "error": f"function execution failed: {e}"}

        return {
            "ok": True,
            "kind": "code",
            "result": result,
        }
