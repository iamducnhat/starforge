from __future__ import annotations

import math
import re
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
        self.embedding_dims = max(64, min(int(embedding_dims), 2048))
        ensure_dir(self.blocks_dir)
        self.stats_path = self.blocks_dir / "_stats.json"
        self._metadata_cache: dict[str, dict[str, Any]] = {}
        self._stats_cache: dict[str, dict[str, Any]] = {}
        self._load_stats()
        self._load_metadata()  # Initial load

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
            bucket = hash(token) % self.embedding_dims
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
