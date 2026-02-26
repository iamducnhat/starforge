from __future__ import annotations

import ast
import hashlib
import io
import tokenize
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .utils import ensure_dir, normalize_keywords, read_json, slugify, utc_now_iso, write_json, write_text


class FunctionRegistry:
    def __init__(self, functions_dir: str | Path = "functions") -> None:
        self.functions_dir = Path(functions_dir)
        ensure_dir(self.functions_dir)

    def _metadata_files(self):
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
        code: str,
        keywords: list[str],
        version: str = "1.0.0",
    ) -> dict[str, Any]:
        safe_name = slugify(name)
        py_path = self.functions_dir / f"{safe_name}.py"
        meta_path = self.functions_dir / f"{safe_name}.json"

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
        }

        write_text(py_path, code_text)
        write_json(meta_path, metadata)

        return {
            "ok": True,
            "created": True,
            "function_file": str(py_path),
            "metadata_file": str(meta_path),
            "code_hash": code_hash,
        }
