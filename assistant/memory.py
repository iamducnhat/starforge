from __future__ import annotations

import json
import math
import re
from hashlib import blake2b
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .utils import (ensure_dir, normalize_keywords, read_json, slugify,
                    utc_now_iso, write_json, write_text)


class MemoryStore:
    def __init__(
        self, blocks_dir: str | Path = "memory/blocks", embedding_dims: int = 256
    ) -> None:
        self.blocks_dir = Path(blocks_dir)
        if self.blocks_dir.name == "blocks":
            self.strategies_dir = self.blocks_dir.parent / "strategies"
        else:
            self.strategies_dir = self.blocks_dir.parent / f"{self.blocks_dir.name}_strategies"
        self.root_causes_dir = self.blocks_dir.parent / "root_causes"
        self.embedding_dims = max(64, min(int(embedding_dims), 2048))
        ensure_dir(self.blocks_dir)
        ensure_dir(self.strategies_dir)
        ensure_dir(self.root_causes_dir)
        self.stats_path = self.blocks_dir / "_stats.json"
        self._metadata_cache: dict[str, dict[str, Any]] = {}
        self._stats_cache: dict[str, dict[str, Any]] = {}
        self._strategy_cache: dict[str, dict[str, Any]] = {}
        self._root_cause_cache: dict[str, list[dict[str, Any]]] = {}
        self._root_cause_index: dict[str, tuple[Path, int]] = {}
        self._root_cause_seed_files = ("import_errors.json", "test_failures.json")
        self._load_stats()
        self._load_metadata()  # Initial load
        self._load_strategies()
        self._load_root_causes()

    def _load_stats(self) -> None:
        try:
            data = read_json(self.stats_path)
            if isinstance(data, dict):
                self._stats_cache = {
                    str(k): v for k, v in data.items() if isinstance(v, dict)
                }
            else:
                self._stats_cache = {}
        except Exception:
            self._stats_cache = {}

    def _save_stats(self) -> None:
        try:
            write_json(self.stats_path, self._stats_cache)
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

    def _recency_bonus(self, created_at: str, last_success_at: str) -> float:
        now = datetime.now(timezone.utc)
        base = self._parse_iso_to_utc(last_success_at) or self._parse_iso_to_utc(created_at)
        if base is None:
            return 0.0
        age_days = max(0.0, (now - base).total_seconds() / 86400.0)
        # 0.0 .. 0.45
        return 0.45 / (1.0 + (age_days / 30.0))

    def _success_bonus(self, block_name: str) -> float:
        stats = self._stats_cache.get(block_name, {})
        success = float(stats.get("success_count", 0.0) or 0.0)
        failure = float(stats.get("failure_count", 0.0) or 0.0)
        confidence_sum = float(stats.get("confidence_sum", 0.0) or 0.0)
        uses = max(1.0, success + failure)
        confidence_avg = confidence_sum / uses
        net = max(0.0, success - (failure * 0.5))
        # 0.0 .. ~1.1
        return min(1.1, (net * 0.08) + (confidence_avg * 0.35))

    def _register_usage(self, block_name: str) -> None:
        stats = self._stats_cache.setdefault(block_name, {})
        stats["use_count"] = int(stats.get("use_count", 0) or 0) + 1
        stats["last_used_at"] = utc_now_iso()
        self._save_stats()

    def record_feedback(
        self,
        block_name: str,
        success: bool,
        confidence: float = 1.0,
        source: str = "runtime",
    ) -> dict[str, Any]:
        slug = slugify(block_name)
        stats = self._stats_cache.setdefault(slug, {})
        if success:
            stats["success_count"] = int(stats.get("success_count", 0) or 0) + 1
            stats["last_success_at"] = utc_now_iso()
        else:
            stats["failure_count"] = int(stats.get("failure_count", 0) or 0) + 1
            stats["last_failure_at"] = utc_now_iso()
        conf = max(0.0, min(float(confidence), 1.0))
        stats["confidence_sum"] = float(stats.get("confidence_sum", 0.0) or 0.0) + conf
        stats["feedback_count"] = int(stats.get("feedback_count", 0) or 0) + 1
        stats["feedback_source"] = str(source or "runtime")
        self._save_stats()
        return {"ok": True, "block": slug, "stats": stats}

    def _build_embedding_source(
        self, info: dict[str, Any], knowledge: str = ""
    ) -> str:
        parts: list[str] = []
        for key in ("name", "topic"):
            value = str(info.get(key, "")).strip()
            if value:
                parts.append(value)
        keywords = info.get("keywords", [])
        if isinstance(keywords, list):
            parts.extend(str(k).strip() for k in keywords if str(k).strip())
        if knowledge:
            parts.append(knowledge[:12000])
        return "\n".join(parts)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]{2,}", text.lower())

    def _embed_text(self, text: str) -> dict[int, float]:
        tokens = self._tokenize(text)
        if not tokens:
            return {}

        counts: dict[int, float] = {}
        for token in tokens:
            # Use deterministic hashing so semantic retrieval is stable across
            # processes/runs (Python's built-in hash is randomized per process).
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "little") % self.embedding_dims
            counts[bucket] = counts.get(bucket, 0.0) + 1.0

        norm = math.sqrt(sum(v * v for v in counts.values()))
        if norm <= 0:
            return {}
        return {k: (v / norm) for k, v in counts.items()}

    @staticmethod
    def _cosine_sparse(a: dict[int, float], b: dict[int, float]) -> float:
        if not a or not b:
            return 0.0
        if len(a) > len(b):
            a, b = b, a
        score = 0.0
        for k, v in a.items():
            score += v * b.get(k, 0.0)
        return float(score)

    def _render_result(
        self,
        info: dict[str, Any],
        *,
        block_name: str,
        score: float,
        overlap: list[str] | None = None,
        semantic_score: float = 0.0,
        match_type: str = "keyword",
    ) -> dict[str, Any]:
        knowledge = str(info.get("_knowledge", "")).strip()
        if not knowledge:
            knowledge_path = info["_knowledge_path"]
            try:
                knowledge = knowledge_path.read_text(encoding="utf-8").strip()
            except Exception:
                knowledge = "[Error reading knowledge content]"

        return {
            "score": round(score, 3),
            "semantic_score": round(semantic_score, 3),
            "match_type": match_type,
            "overlap": sorted(overlap or []),
            "block": block_name,
            "name": info.get("name", info["_block_dir"].name),
            "topic": info.get("topic", ""),
            "keywords": info.get("keywords", []),
            "dependencies": info.get("dependencies", []),
            "source": info.get("source", []),
            "version": info.get("version", "1.0.0"),
            "created_at": info.get("created_at", ""),
            "knowledge": knowledge,
        }

    def _load_metadata(self) -> None:
        """Load or refresh the metadata cache from disk."""
        new_cache = {}
        for block_dir, info_path, knowledge_path in self._iter_blocks():
            try:
                info = read_json(info_path)
                try:
                    knowledge_text = knowledge_path.read_text(encoding="utf-8").strip()
                except Exception:
                    knowledge_text = ""
                # Store paths and normalized keywords for fast lookup
                info["_block_dir"] = block_dir
                info["_knowledge_path"] = knowledge_path
                info["_norm_keywords"] = set(
                    normalize_keywords(info.get("keywords", []))
                )
                info["_embedding"] = self._embed_text(
                    self._build_embedding_source(info, knowledge_text)
                )
                info["_knowledge"] = knowledge_text
                new_cache[block_dir.name] = info
            except Exception:
                continue
        self._metadata_cache = new_cache

    def _iter_blocks(self) -> Iterator[tuple[Path, Path, Path]]:
        """Iterate over valid memory blocks, yielding (block_dir, info_path, knowledge_path)."""
        for block_dir in sorted(self.blocks_dir.glob("*")):
            if not block_dir.is_dir():
                continue
            info_path = block_dir / "info.json"
            knowledge_path = block_dir / "knowledge.md"
            if info_path.exists() and knowledge_path.exists():
                yield block_dir, info_path, knowledge_path

    def _strategy_path(self, strategy_id: str) -> Path:
        return self.strategies_dir / f"{strategy_id}.json"

    def _root_cause_paths(self) -> list[Path]:
        paths: list[Path] = []
        for name in self._root_cause_seed_files:
            path = self.root_causes_dir / name
            if not path.exists():
                write_json(path, [])
            paths.append(path)
        for path in sorted(self.root_causes_dir.glob("*.json")):
            if path not in paths:
                paths.append(path)
        return paths

    def _load_root_causes(self) -> None:
        cache: dict[str, list[dict[str, Any]]] = {}
        index: dict[str, tuple[Path, int]] = {}
        for path in self._root_cause_paths():
            try:
                payload = read_json(path)
            except Exception:
                payload = []
            if not isinstance(payload, list):
                payload = []
            normalized: list[dict[str, Any]] = []
            for idx, raw_entry in enumerate(payload):
                entry = self._normalize_root_cause_entry(raw_entry, path=path, idx=idx)
                if not entry:
                    continue
                normalized.append(entry)
                index[entry["id"]] = (path, len(normalized) - 1)
            cache[path.name] = normalized
        self._root_cause_cache = cache
        self._root_cause_index = index

    @staticmethod
    def _normalize_root_cause_entry(
        raw_entry: Any, *, path: Path, idx: int
    ) -> dict[str, Any] | None:
        if not isinstance(raw_entry, dict):
            return None
        pattern = str(raw_entry.get("pattern", "")).strip()
        if not pattern:
            return None
        context = raw_entry.get("context", {})
        if not isinstance(context, dict):
            context = {}
        fix_template = raw_entry.get("fix_template", [])
        if not isinstance(fix_template, list):
            fix_template = []
        key_basis = json.dumps(
            {"pattern": pattern, "context": context, "fix_template": fix_template},
            ensure_ascii=False,
            sort_keys=True,
        )
        entry_id = str(raw_entry.get("id", "")).strip()
        if not entry_id:
            entry_id = f"{path.stem}_{blake2b(key_basis.encode('utf-8'), digest_size=5).hexdigest()}"
        try:
            confidence = float(raw_entry.get("confidence", 0.5))
        except Exception:
            confidence = 0.5
        return {
            "id": entry_id,
            "pattern": pattern,
            "context": context,
            "fix_template": fix_template,
            "success_count": int(raw_entry.get("success_count", 0) or 0),
            "fail_count": int(raw_entry.get("fail_count", 0) or 0),
            "confidence": max(0.0, min(confidence, 1.0)),
        }

    @staticmethod
    def _placeholder_pattern_to_regex(pattern: str) -> tuple[str, list[str]]:
        placeholders = re.findall(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}", pattern)
        escaped = re.escape(pattern)
        for name in placeholders:
            escaped = escaped.replace(
                re.escape("${" + name + "}"),
                rf"(?P<{name}>[a-zA-Z0-9_.\-\/]+)",
            )
        return escaped, placeholders

    @staticmethod
    def _root_cause_context_match(
        expected: dict[str, Any], runtime: dict[str, Any]
    ) -> float:
        if not expected:
            return 0.2
        matched = 0
        total = 0
        for key, value in expected.items():
            total += 1
            if str(runtime.get(key, "")).strip().lower() == str(value).strip().lower():
                matched += 1
        if total <= 0:
            return 0.2
        return matched / total

    @classmethod
    def _match_root_cause_pattern(
        cls, entry: dict[str, Any], error_text: str
    ) -> tuple[bool, dict[str, str], float]:
        pattern = str(entry.get("pattern", "")).strip()
        if not pattern or not error_text:
            return False, {}, 0.0
        captures: dict[str, str] = {}
        text = error_text.strip()
        lowered = text.lower()

        if pattern.startswith("re:"):
            raw = pattern[3:].strip()
            if not raw:
                return False, {}, 0.0
            m = re.search(raw, text, re.IGNORECASE | re.DOTALL)
            if not m:
                return False, {}, 0.0
            captures = {
                str(k): str(v).strip()
                for k, v in m.groupdict().items()
                if v is not None and str(v).strip()
            }
            return True, captures, min(1.0, 0.55 + (len(m.group(0)) / max(60.0, len(text))))

        regex, placeholders = cls._placeholder_pattern_to_regex(pattern)
        m = re.search(regex, text, re.IGNORECASE)
        if m:
            for name in placeholders:
                value = m.groupdict().get(name, "")
                if value:
                    captures[name] = str(value).strip()
            return True, captures, min(1.0, 0.65 + (len(m.group(0)) / max(80.0, len(text))))

        keywords = re.findall(r"[a-zA-Z0-9_]{3,}", pattern.lower())
        if not keywords:
            return False, {}, 0.0
        overlap = sum(1 for kw in keywords if kw in lowered)
        if overlap <= 0:
            return False, {}, 0.0
        keyword_score = overlap / max(1.0, len(keywords))
        return True, {}, keyword_score * 0.6

    @classmethod
    def _interpolate_root_cause_template(
        cls, value: Any, captures: dict[str, str]
    ) -> Any:
        if isinstance(value, str):
            out = value
            for key, captured in captures.items():
                out = out.replace("${" + key + "}", str(captured))
            return out
        if isinstance(value, dict):
            return {
                str(k): cls._interpolate_root_cause_template(v, captures)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [cls._interpolate_root_cause_template(v, captures) for v in value]
        return value

    @staticmethod
    def _normalize_strategy_step(step: Any, fallback_id: int) -> dict[str, Any]:
        if not isinstance(step, dict):
            return {
                "step_id": fallback_id,
                "action": str(step).strip() or f"step_{fallback_id}",
                "args": {},
                "depends_on": [fallback_id - 1] if fallback_id > 1 else [],
                "expected_output": "",
            }

        raw_step_id = step.get("step_id", step.get("id", fallback_id))
        try:
            step_id = int(raw_step_id)
        except Exception:
            step_id = fallback_id
        if step_id <= 0:
            step_id = fallback_id

        args = step.get("args", {})
        if not isinstance(args, dict):
            args = {}

        deps: list[int] = []
        raw_deps = step.get("depends_on", [])
        if isinstance(raw_deps, list):
            for item in raw_deps:
                try:
                    dep = int(item)
                except Exception:
                    continue
                if dep > 0 and dep not in deps and dep != step_id:
                    deps.append(dep)

        return {
            "step_id": step_id,
            "action": str(step.get("action", "")).strip() or f"step_{step_id}",
            "args": args,
            "depends_on": deps,
            "expected_output": str(
                step.get("expected_output", "") or step.get("expected", "")
            ).strip(),
        }

    def _strategy_embedding_source(self, strategy: dict[str, Any]) -> str:
        parts = [str(strategy.get("goal", "")).strip()]
        for step in strategy.get("strategy", []):
            if not isinstance(step, dict):
                continue
            action = str(step.get("action", "")).strip()
            expected = str(step.get("expected_output", "")).strip()
            if action:
                parts.append(action)
            if expected:
                parts.append(expected)
            args = step.get("args", {})
            if isinstance(args, dict):
                for key, value in args.items():
                    parts.append(f"{key} {value}")
        notes = str(strategy.get("notes", "")).strip()
        if notes:
            parts.append(notes)
        return "\n".join(part for part in parts if part)

    def _load_strategies(self) -> None:
        cache: dict[str, dict[str, Any]] = {}
        for path in sorted(self.strategies_dir.glob("*.json")):
            try:
                item = read_json(path)
            except Exception:
                continue
            if not isinstance(item, dict):
                continue
            strategy_id = str(item.get("id", path.stem)).strip() or path.stem
            item["id"] = strategy_id
            item["_path"] = str(path)
            item["_embedding"] = self._embed_text(self._strategy_embedding_source(item))
            cache[strategy_id] = item
        self._strategy_cache = cache

    def record_strategy(
        self,
        goal: str,
        strategy: list[dict[str, Any]] | list[Any],
        success: bool,
        source: str = "runtime",
        notes: str = "",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_goal = str(goal or "").strip()
        if not normalized_goal:
            return {"ok": False, "error": "goal must not be empty"}

        normalized_steps = [
            self._normalize_strategy_step(step, idx)
            for idx, step in enumerate(strategy or [], start=1)
        ]
        if not normalized_steps:
            return {"ok": False, "error": "strategy must include at least one step"}

        fingerprint_basis = json.dumps(
            {"goal": normalized_goal, "strategy": normalized_steps},
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = blake2b(
            fingerprint_basis.encode("utf-8"), digest_size=6
        ).hexdigest()
        base = slugify(normalized_goal) or "strategy"
        strategy_id = f"{base[:48]}_{digest}"
        path = self._strategy_path(strategy_id)

        existing: dict[str, Any] = {}
        if path.exists():
            try:
                loaded = read_json(path)
                if isinstance(loaded, dict):
                    existing = loaded
            except Exception:
                existing = {}

        entry = {
            "id": strategy_id,
            "goal": normalized_goal,
            "strategy": normalized_steps,
            "context": context if isinstance(context, dict) else existing.get("context", {}),
            "created_at": existing.get("created_at", utc_now_iso()),
            "updated_at": utc_now_iso(),
            "success_count": int(existing.get("success_count", 0) or 0),
            "failure_count": int(existing.get("failure_count", 0) or 0),
            "last_used_at": utc_now_iso(),
            "last_success_at": existing.get("last_success_at", ""),
            "source": str(source or existing.get("source", "runtime")),
            "notes": str(notes or existing.get("notes", "")).strip()[:800],
        }
        if success:
            entry["success_count"] += 1
            entry["last_success_at"] = utc_now_iso()
        else:
            entry["failure_count"] += 1

        write_json(path, entry)
        entry["_path"] = str(path)
        entry["_embedding"] = self._embed_text(self._strategy_embedding_source(entry))
        self._strategy_cache[strategy_id] = entry
        return {"ok": True, "strategy": entry}

    def find_strategies(
        self, query: str, limit: int = 5, min_score: float = 0.15
    ) -> list[dict[str, Any]]:
        query_text = str(query or "").strip()
        if not query_text:
            return []

        query_embedding = self._embed_text(query_text)
        if not query_embedding:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self._strategy_cache.values():
            semantic = self._cosine_sparse(query_embedding, item.get("_embedding", {}))
            success_count = float(item.get("success_count", 0.0) or 0.0)
            failure_count = float(item.get("failure_count", 0.0) or 0.0)
            total = max(1.0, success_count + failure_count)
            success_ratio = success_count / total
            score = (
                semantic
                + (success_ratio * 0.45)
                + self._recency_bonus(
                    str(item.get("created_at", "")),
                    str(item.get("last_success_at", "")),
                )
            )
            if score < min_score:
                continue
            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        out: list[dict[str, Any]] = []
        for score, item in scored[: max(1, limit)]:
            out.append(
                {
                    "id": item.get("id", ""),
                    "goal": item.get("goal", ""),
                    "strategy": item.get("strategy", []),
                    "context": item.get("context", {}),
                    "success_count": int(item.get("success_count", 0) or 0),
                    "failure_count": int(item.get("failure_count", 0) or 0),
                    "success_rate": round(success_ratio, 3),
                    "last_success_at": item.get("last_success_at", ""),
                    "score": round(score, 3),
                }
            )
        return out

    def find_root_causes(
        self,
        error_text: str,
        context: dict[str, Any] | None = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        text = str(error_text or "").strip()
        if not text:
            return []
        runtime_context = context if isinstance(context, dict) else {}
        candidates: list[dict[str, Any]] = []
        for entries in self._root_cause_cache.values():
            for entry in entries:
                matched, captures, pattern_score = self._match_root_cause_pattern(entry, text)
                if not matched:
                    continue
                context_score = self._root_cause_context_match(
                    entry.get("context", {}), runtime_context
                )
                if context_score <= 0.0:
                    continue
                success = float(entry.get("success_count", 0.0) or 0.0)
                failures = float(entry.get("fail_count", 0.0) or 0.0)
                observed = max(1.0, success + failures)
                success_ratio = success / observed
                confidence = float(entry.get("confidence", 0.0) or 0.0)
                score = (
                    (pattern_score * 0.55)
                    + (context_score * 0.2)
                    + (confidence * 0.2)
                    + (success_ratio * 0.15)
                )
                interpolated = self._interpolate_root_cause_template(
                    entry.get("fix_template", []), captures
                )
                candidates.append(
                    {
                        "id": entry.get("id", ""),
                        "pattern": entry.get("pattern", ""),
                        "context": entry.get("context", {}),
                        "captures": captures,
                        "fix_template": interpolated if isinstance(interpolated, list) else [],
                        "confidence": round(confidence, 3),
                        "success_count": int(success),
                        "fail_count": int(failures),
                        "score": round(score, 3),
                    }
                )
        candidates.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        return candidates[: max(1, limit)]

    def record_root_cause_feedback(
        self, root_cause_id: str, success: bool, confidence: float = 1.0
    ) -> dict[str, Any]:
        root_id = str(root_cause_id or "").strip()
        if not root_id:
            return {"ok": False, "error": "root_cause_id is required"}
        located = self._root_cause_index.get(root_id)
        if located is None:
            return {"ok": False, "error": f"unknown root cause id: {root_id}"}
        path, idx = located
        entries = self._root_cause_cache.get(path.name, [])
        if idx < 0 or idx >= len(entries):
            return {"ok": False, "error": f"root cause not found in cache: {root_id}"}
        entry = entries[idx]
        if success:
            entry["success_count"] = int(entry.get("success_count", 0) or 0) + 1
        else:
            entry["fail_count"] = int(entry.get("fail_count", 0) or 0) + 1
        try:
            conf = float(confidence)
        except Exception:
            conf = 1.0
        conf = max(0.0, min(conf, 1.0))
        current = float(entry.get("confidence", 0.5) or 0.5)
        entry["confidence"] = round((current * 0.8) + (conf * 0.2), 4)
        serializable = [
            {
                "id": item.get("id", ""),
                "pattern": item.get("pattern", ""),
                "context": item.get("context", {}),
                "fix_template": item.get("fix_template", []),
                "success_count": int(item.get("success_count", 0) or 0),
                "fail_count": int(item.get("fail_count", 0) or 0),
                "confidence": float(item.get("confidence", 0.0) or 0.0),
            }
            for item in entries
        ]
        write_json(path, serializable)
        return {"ok": True, "id": root_id, "entry": entry}

    def semantic_search(
        self, query: str, limit: int = 5, min_score: float = 0.1
    ) -> list[dict[str, Any]]:
        query_text = (query or "").strip()
        if not query_text:
            return []

        query_embedding = self._embed_text(query_text)
        if not query_embedding:
            return []

        scored: list[tuple[float, float, str, dict[str, Any]]] = []
        for block_name, info in self._metadata_cache.items():
            semantic = self._cosine_sparse(query_embedding, info.get("_embedding", {}))
            score = semantic + self._success_bonus(block_name) + self._recency_bonus(
                str(info.get("created_at", "")),
                str(self._stats_cache.get(block_name, {}).get("last_success_at", "")),
            )
            if score < min_score:
                continue
            scored.append((score, semantic, block_name, info))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[dict[str, Any]] = []
        for score, semantic, block_name, info in scored[: max(1, limit)]:
            self._register_usage(block_name)
            out.append(
                self._render_result(
                    info,
                    block_name=block_name,
                    score=score,
                    semantic_score=semantic,
                    match_type="semantic",
                )
            )
        return out

    def find_in_memory(
        self, keywords: list[str], limit: int = 5
    ) -> list[dict[str, Any]]:
        """
        Search for memory blocks matching the given keywords.

        Combines keyword overlap with semantic similarity over local embeddings.
        """
        requested = set(normalize_keywords(keywords))
        if not requested:
            return []

        query_text = " ".join(sorted(requested))
        query_embedding = self._embed_text(query_text)
        scored_matches: list[dict[str, Any]] = []
        for block_name, info in self._metadata_cache.items():
            stored = info.get("_norm_keywords", set())
            overlap = requested & stored

            overlap_score = len(overlap)
            recall_score = len(overlap) / max(1, len(requested))
            semantic_score = self._cosine_sparse(
                query_embedding, info.get("_embedding", {})
            )

            if not overlap and semantic_score < 0.1:
                continue

            # Weighted blend: keep lexical precision while allowing semantic recall.
            score = (
                overlap_score
                + recall_score
                + (semantic_score * 1.75)
                + self._success_bonus(block_name)
                + self._recency_bonus(
                    str(info.get("created_at", "")),
                    str(self._stats_cache.get(block_name, {}).get("last_success_at", "")),
                )
            )
            match_type = "hybrid" if overlap else "semantic"

            scored_matches.append(
                {
                    "block_name": block_name,
                    "score": round(score, 3),
                    "semantic_score": round(semantic_score, 3),
                    "match_type": match_type,
                    "overlap": sorted(overlap),
                    "info": info,
                }
            )

        # Sort by score and overlap size
        scored_matches.sort(key=lambda x: (x["score"], len(x["overlap"])), reverse=True)

        results: list[dict[str, Any]] = []
        for match in scored_matches[:limit]:
            info = match["info"]
            block_name = str(match.get("block_name", ""))
            self._register_usage(block_name)
            results.append(
                self._render_result(
                    info,
                    block_name=block_name,
                    score=match["score"],
                    overlap=match["overlap"],
                    semantic_score=match["semantic_score"],
                    match_type=match["match_type"],
                )
            )

        return results

    def create_block(
        self,
        name: str,
        topic: str,
        keywords: list[str],
        knowledge: str,
        source: list[str],
        dependencies: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new memory block with the given metadata and knowledge content.

        Args:
            name: Human-readable name of the block.
            topic: General topic area.
            keywords: List of keywords for retrieval.
            knowledge: Markdown content of the knowledge block.
            source: List of source URLs or references.
            dependencies: Optional list of other block names this depends on.
        """
        block_name = slugify(name)
        block_dir = self.blocks_dir / block_name
        ensure_dir(block_dir)

        info = {
            "name": name,
            "topic": topic,
            "keywords": normalize_keywords(keywords),
            "dependencies": dependencies or [],
            "source": source,
            "version": "1.0.0",
            "created_at": utc_now_iso(),
        }

        info_path = block_dir / "info.json"
        knowledge_path = block_dir / "knowledge.md"

        write_json(info_path, info)
        write_text(knowledge_path, knowledge.strip() + "\n")

        # Update cache
        info_copy = info.copy()
        info_copy["_block_dir"] = block_dir
        info_copy["_knowledge_path"] = knowledge_path
        info_copy["_norm_keywords"] = set(info["keywords"])
        info_copy["_embedding"] = self._embed_text(
            self._build_embedding_source(info, knowledge.strip())
        )
        info_copy["_knowledge"] = knowledge.strip()
        self._metadata_cache[block_name] = info_copy

        return {
            "ok": True,
            "block": block_name,
            "info_path": str(info_path),
            "knowledge_path": str(knowledge_path),
        }
